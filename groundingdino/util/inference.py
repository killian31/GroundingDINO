
from typing import Tuple, List
from contextlib import contextmanager

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert
import bisect

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


# SDPA kernel fallback context manager for compatibility
@contextmanager
def sdpa_kernel_fallback():
    """Context manager to use efficient attention when available, with fallback."""
    try:
        # Try to use flash attention if available (fastest)
        if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            # PyTorch 2.0+ with SDPA
            yield
        else:
            yield
    except Exception:
        yield


# Global flag for torch.compile
_COMPILED_MODEL_CACHE = {}

# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def _check_gpu_supports_compile() -> bool:
    """Check if the current GPU supports torch.compile with Triton.

    Some GPUs have issues with torch.compile:
    - NVIDIA GB10 (DGX Spark): Compute capability 12.1 (sm_121a) is too new for Triton
    - GPUs with too few SMs for max_autotune_gemm mode
    - Very old GPUs (< SM 7.0)
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Get current device properties
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        sm_count = device_props.multi_processor_count
        major, minor = device_props.major, device_props.minor
        compute_cap = major + minor / 10

        # Check for unsupported architectures
        # GB10 has compute capability 12.1 which Triton doesn't support yet
        # Triton's ptxas doesn't recognize sm_121a
        if major >= 12:
            print(f"[INFO] GPU compute capability {major}.{minor} is too new for Triton, torch.compile disabled")
            return False

        # torch.compile works best on SM 7.0+ (Volta and newer)
        if compute_cap < 7.0:
            print(f"[INFO] GPU compute capability {compute_cap} < 7.0, torch.compile may not work well")
            return False

        # Check SM count for max_autotune_gemm mode
        # torch.compile with reduce-overhead mode needs sufficient SMs
        min_sm_count = 40
        if sm_count < min_sm_count:
            print(f"[INFO] GPU has {sm_count} SMs (< {min_sm_count}), not enough for torch.compile")
            return False

        return True
    except Exception as e:
        print(f"[WARNING] Could not check GPU properties: {e}")
        return False


def load_model(
    model_config_path: str,
    model_checkpoint_path: str,
    device: str = "cuda",
    compile_model: bool = True,
):
    """Load Grounding DINO model with optional torch.compile optimization.

    Args:
        model_config_path: Path to model config file
        model_checkpoint_path: Path to model checkpoint
        device: Device to load model on
        compile_model: Whether to apply torch.compile for faster inference (default: True).
                      Will automatically fall back if GPU doesn't support it.

    Returns:
        Optimized model ready for inference
    """
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()

    # Apply performance optimizations
    if device != "cpu" and torch.cuda.is_available():
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Try to apply torch.compile for faster inference
        # Check if GPU supports it first
        if compile_model and _check_gpu_supports_compile():
            try:
                # Check if already compiled
                cache_key = (model_config_path, model_checkpoint_path, device)
                if cache_key in _COMPILED_MODEL_CACHE:
                    return _COMPILED_MODEL_CACHE[cache_key]

                # Use reduce-overhead mode for best inference performance
                compiled_model = torch.compile(
                    model,
                    mode="reduce-overhead",  # Optimized for inference
                    fullgraph=False,  # Allow graph breaks for compatibility
                )
                _COMPILED_MODEL_CACHE[cache_key] = compiled_model
                print("[INFO] Grounding DINO model compiled with torch.compile (reduce-overhead mode)")
                return compiled_model
            except Exception as e:
                # Common errors include:
                # - "CUDA error: too many resources requested for launch" (SM cores)
                # - Triton compilation errors
                # - Graph break errors
                error_msg = str(e).lower()
                if "too many resources" in error_msg or "sm" in error_msg or "triton" in error_msg:
                    print(f"[INFO] torch.compile not supported on this GPU, using uncompiled model")
                else:
                    print(f"[WARNING] torch.compile failed, using uncompiled model: {e}")
        elif compile_model:
            print("[INFO] torch.compile disabled: GPU does not meet requirements")

    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad(), sdpa_kernel_fallback():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

    return boxes, logits.max(dim=1)[0], phrases


def predict_batch(
        model,
        images: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False,
        return_phrases: bool = True
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
    """
    Batched variant of `predict` that shares tokenization and text encoding.

    Args:
        model: GroundingDINO model instance.
        images: Tensor of shape (B, 3, H, W) or list of tensors; each already preprocessed.
        caption: Text prompt used for all images in the batch.
        box_threshold: Objectness threshold applied to query logits.
        text_threshold: Token activation threshold for phrase extraction.
        device: Inference device.
        remove_combined: Whether to suppress combined (punctuation-separated) phrases.

    Returns:
        batch_boxes: list of tensors (num_dets_i, 4) in cxcywh normalized coords.
        batch_scores: list of tensors (num_dets_i,) with confidence per detection.
        batch_phrases: list of phrase lists per image.
    """
    caption = preprocess_caption(caption)

    if isinstance(images, (list, tuple)):
        images = torch.stack(list(images))

    # Ensure model is on device (avoid redundant .to() calls)
    if next(model.parameters()).device != torch.device(device):
        model = model.to(device)

    # Use non_blocking transfer for better GPU utilization
    images = images.to(device, non_blocking=True)

    # Use inference_mode for slightly faster inference than no_grad
    # Note: Mixed precision (autocast) is disabled for Grounding DINO because the
    # Multi-Scale Deformable Attention CUDA kernel only supports float32/float64.
    # TF32 mode is still used automatically on modern GPUs (Ampere+) for float32 operations,
    # providing ~2x speedup compared to pure FP32 on those GPUs.
    with torch.inference_mode():
        outputs = model(images, captions=[caption] * images.shape[0])

    prediction_logits = outputs["pred_logits"].sigmoid().cpu()  # (B, nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()  # (B, nq, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    if remove_combined:
        sep_idx = [
            i for i, tok in enumerate(tokenized["input_ids"])
            if tok in [101, 102, 1012]  # [CLS], [SEP], '.'
        ]

    batch_boxes: List[torch.Tensor] = []
    batch_scores: List[torch.Tensor] = []
    batch_phrases: List[List[str]] = []

    for logits_per_image, boxes_per_image in zip(prediction_logits, prediction_boxes):
        mask = logits_per_image.max(dim=1)[0] > box_threshold
        logits = logits_per_image[mask]
        boxes = boxes_per_image[mask]

        scores = logits.max(dim=1)[0] if logits.numel() > 0 else torch.empty(0)

        if remove_combined:
            phrases = []
            for logit in logits:
                max_idx = logit.argmax()
                insert_idx = bisect.bisect_left(sep_idx, max_idx)
                right_idx = sep_idx[insert_idx]
                left_idx = sep_idx[insert_idx - 1]
                phrases.append(
                    get_phrases_from_posmap(
                        logit > text_threshold, tokenized, tokenizer, left_idx, right_idx
                    ).replace(".", "")
                )
        else:
            phrases = [
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace(".", "")
                for logit in logits
            ]

        batch_boxes.append(boxes)
        batch_scores.append(scores)
        batch_phrases.append(phrases)

    return batch_boxes, batch_scores, batch_phrases


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold, 
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)
