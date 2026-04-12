import json
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from .model import TabDPTModel
from .utils import FAISS, convert_to_torch_tensor, Log1pScaler, generate_random_permutation
from typing import Union


class _AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _to_attr_dict(obj):
    if isinstance(obj, dict):
        return _AttrDict({k: _to_attr_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(v) for v in obj]
    return obj

# Constants for model caching and download
_VERSION = "1_1"
_MODEL_NAME = f"tabdpt{_VERSION}.safetensors"
_HF_REPO_ID = "Layer6/TabDPT"
CPU_INF_BATCH = 16


def _is_cuda_device(device: str | None) -> bool:
    return isinstance(device, str) and device.startswith("cuda")


class TabDPTEstimator(BaseEstimator):
    @staticmethod
    def download_weights() -> str:
        path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_MODEL_NAME,
        )
        return path

    def __init__(
        self,
        mode: Literal["cls", "reg"],
        inf_batch_size: int = 512,
        normalizer: Literal["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p"] | None
            = "standard",
        missing_indicators: bool = False,
        clip_sigma: float = 4.,
        feature_reduction: Literal["pca", "subsample"] = "pca",
        faiss_metric: Literal["l2", "ip"] = "l2",
        device: str = None,
        use_flash: bool = True,
        compile: bool = True,
        model_weight_path: str | None = None,
        text_enhanced: bool = False,
        text_attn_layers: list[int] | None = None,
    ):
        """
        Initializes the TabDPT Estimator
        Args:
            mode: Defines what mode the estimator is
                "cls" is classification, "reg" is regression
            inf_batch_size: The batch size for inferencing
            normalizer: Specifies normalization used for preprocessing before retrieval. Note that
                the model performs additional normalization in its forward function. By default the
                scikit-learn StandardScaler is used, which matches model training. Other options are:
                - "minmax": scikit-learn MinMaxScaler(feature_range=(-1,1))
                - "robust": scikit-learn RobustScaler()
                - "power": scikit-learn PowerTransformer()
                - "quantile-uniform": scikit-learn QuantileTransformer(output_distribution="uniform"), rescaled to (-1,1)
                - "quantile-normal": scikit-learn QuantileTransformer(output_distribution="normal")
                - "log1p": sign(X) * log(1 + abs(X))
                - None: no normalization
            missing_indicators: If True, adds an additional binary column for each feature with
                missing values indicating their position.
            clip_sigma: n*sigma used for outlier clipping
            feature_reduction: Method used to reduce the number of features when over the model's
                limit, either "pca" or "subsample"
            faiss_metric: Distance used for retrieval, either "l2" or "ip"
            device: Specifies the computational device (e.g., CPU, GPU)
                Identical to https://docs.pytorch.org/docs/stable/generated/torch.cuda.device.html
            use_flash: Specifies whether to use flash attention or not
            compile: Specifies whether to compile the model with torch before inference
            model_weight_path: path on file system specifying the model weights
                If no path is specified, then the model weights are downloaded from HuggingFace

        """
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inf_batch_size = inf_batch_size if _is_cuda_device(self.device) else min(inf_batch_size, CPU_INF_BATCH)
        self.use_flash = use_flash and _is_cuda_device(self.device)
        self.missing_indicators = missing_indicators
        self.text_attn_layers = text_attn_layers
        self.text_enhanced = text_enhanced or text_attn_layers is not None

        if model_weight_path:
            self.path = model_weight_path
        else:
            self.path = self.download_weights()

        with safe_open(self.path, framework="pt", device=self.device) as f:
            meta = f.metadata()
            cfg_dict = json.loads(meta["cfg"])
            cfg = _to_attr_dict(cfg_dict)
            model_state = {k: f.get_tensor(k) for k in f.keys()}

        cfg.env.device = self.device
        self.model = TabDPTModel.load(
            model_state=model_state,
            config=cfg,
            use_flash=self.use_flash,
            clip_sigma=clip_sigma,
            text_enhanced=self.text_enhanced,
            text_attn_layers=text_attn_layers,
        )
        self.model.eval()


        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out
        self.compile = compile and _is_cuda_device(self.device)
        self.feature_reduction = feature_reduction
        self.faiss_metric = faiss_metric
        assert self.mode in ["cls", "reg"], "mode must be 'cls' or 'reg'"
        assert self.feature_reduction in ["pca", "subsample"], \
                "feature_reduction must be 'pca' or 'subsample'"
        assert self.faiss_metric in ["l2", "ip"], 'faiss_metric must be "l2" or "ip"'

        self.normalizer = normalizer
        match normalizer:
            case "standard":
                self.scaler = StandardScaler()
            case "minmax":
                self.scaler = MinMaxScaler(feature_range=(-1,1))
            case "robust":
                self.scaler = RobustScaler()
            case "power":
                self.scaler = PowerTransformer()
            case "quantile-uniform":
                self.scaler = QuantileTransformer(output_distribution="uniform")
            case "quantile-normal":
                self.scaler = QuantileTransformer(output_distribution="normal")
            case "log1p":
                self.scaler = Log1pScaler()
            case None:
                self.scaler = None
            case _:
                raise ValueError(
                    'normalizer must be one of '
                    '["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p", None]'
                )

    def fit(self, X: np.ndarray, y: np.ndarray, text: np.ndarray | None = None):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"
        if text is not None:
            # text is a 3D array: (N, L, D)
            assert isinstance(text, np.ndarray), "text must be a numpy array"
            assert text.shape[0] == X.shape[0], "text and X must have the same number of samples"
            assert text.ndim == 3, "text must be a 3D array (N, L, D)"

        if self.missing_indicators:
            inds = np.isnan(X)
            self.has_missing_indicator = inds.any(axis=0)
            inds = inds[:, self.has_missing_indicator].astype(float)
            X = np.hstack((X, inds))

        self.imputer = SimpleImputer(strategy="mean")
        X = self.imputer.fit_transform(X)
        if self.scaler:
            X = self.scaler.fit_transform(X)
        if self.normalizer == 'quantile-uniform':
            X = 2*X - 1

        self.faiss_knn = FAISS(X, metric=self.faiss_metric)
        self.n_instances, self.n_features = X.shape
        self.X_train = X
        self.y_train = y
        self.train_text = text
        if self.n_features > self.max_features and self.feature_reduction == "pca":
            train_x = convert_to_torch_tensor(self.X_train).to(self.device).float()
            _, _, self.V = torch.pca_lowrank(train_x, q=min(train_x.shape[0], self.max_features))

        self.is_fitted_ = True
        if self.compile:
            self.model = torch.compile(self.model)

    # text enhancement
    def _compute_pairwise_text_similarity(self, train_text: Union[np.ndarray, torch.Tensor], text_test: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Pairwise cosine similarity between test and train text embeddings, per text feature.
        Automatically handles batch dimension if present.
        Accepts both numpy arrays and torch tensors.
        Returns:
            torch.Tensor shaped (L, N_test, N_train) or (B, L, N_test, N_train) if batched.
        """
        if train_text.shape[-2:] != text_test.shape[-2:]:
            raise ValueError(
                f"Got train={train_text.shape}, test={text_test.shape}."
            )
        # Handle both numpy arrays and tensors properly
        if isinstance(train_text, torch.Tensor):
            train_tensor = train_text.detach().clone().to(dtype=torch.float32, device=self.device)
        else:
            train_tensor = torch.tensor(train_text, dtype=torch.float32, device=self.device)  # (N_train, L, D)
        
        if isinstance(text_test, torch.Tensor):
            test_tensor = text_test.detach().clone().to(dtype=torch.float32, device=self.device)
        else:
            test_tensor = torch.tensor(text_test, dtype=torch.float32, device=self.device)  # (N_test, L, D)
        train_norm = F.normalize(train_tensor, dim=-1)
        test_norm = F.normalize(test_tensor, dim=-1)
        
        # Check if both have batch dimension (4D) or not (3D)
        has_batch = len(train_norm.shape) == 4 and len(test_norm.shape) == 4
        
        if has_batch:
            # Batched version: (B, N_test, L, D) and (B, N_train, L, D) -> (B, L, N_test, N_train)
            # einsum: b=batch, n=N_test, m=N_train, t=L (lag), d=D (embedding dim)
            return torch.einsum("b n t d, b m t d -> b t n m", test_norm, train_norm)
        else:
            # (L, N_test, N_train): cosine similarity for each text feature independently.
            # e.g. result[0, 0, 5] = cosine_sim(test_sample_0_lag1, train_sample_5_lag1)
            return torch.einsum("n t d, m t d -> t n m", test_norm, train_norm) # (L, N_test, N_train)

    def text_embeddings_batched(
        self, train_text: Union[np.ndarray, torch.Tensor], text_test: Union[np.ndarray, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert train/test text to (B, N_train, L, D) and (B, N_test, L, D) on the estimator device.
        Used as `text_train` / `text_test` in TabDPTModel.forward when text_enhanced.
        """
        if isinstance(train_text, torch.Tensor):
            tt = train_text.to(device=self.device, dtype=torch.float32)
        else:
            tt = torch.as_tensor(train_text, dtype=torch.float32, device=self.device)
        if isinstance(text_test, torch.Tensor):
            ts = text_test.to(device=self.device, dtype=torch.float32)
        else:
            ts = torch.as_tensor(text_test, dtype=torch.float32, device=self.device)
        if tt.dim() == 3:
            tt = tt.unsqueeze(0)
        if ts.dim() == 3:
            # KNN path: tt is (B, N_train, L, D) and ts is (B, L, D) with one test row per batch row.
            if tt.dim() == 4 and ts.shape[0] == tt.shape[0]:
                ts = ts.unsqueeze(1)
            else:
                ts = ts.unsqueeze(0)
        return tt, ts

    def _prepare_prediction(self, X: np.ndarray, class_perm: np.ndarray | None = None, seed: int | None = None, text: np.ndarray | None = None):
        check_is_fitted(self)
        
        # Initialize train_text to None at the start
        train_text = None

        if self.missing_indicators:
            inds = np.isnan(X)[:, self.has_missing_indicator].astype(float)
            X = np.hstack((X, inds))
        self.X_test = self.imputer.transform(X)
        if self.scaler:
            self.X_test = self.scaler.transform(self.X_test)
            if self.normalizer == 'quantile-uniform':
                self.X_test = 2*self.X_test - 1

        train_x, train_y, test_x = (
            convert_to_torch_tensor(self.X_train).to(self.device).float(),
            convert_to_torch_tensor(self.y_train).to(self.device).float(),
            convert_to_torch_tensor(self.X_test).to(self.device).float(),
        )

        if self.train_text is not None:
            train_text = convert_to_torch_tensor(self.train_text).to(self.device).float()

        if text is not None:
            text = convert_to_torch_tensor(text).to(self.device).float()

        # Apply PCA/subsampling to reduce the number of features if necessary
        if self.n_features > self.max_features:
            if self.feature_reduction == "pca":
                train_x = train_x @ self.V
                test_x = test_x @ self.V
            elif self.feature_reduction == "subsample":
                feat_perm = generate_random_permutation(train_x.shape[1], seed)
                train_x = train_x[:, feat_perm][:, :self.max_features]
                test_x = test_x[:, feat_perm][:, :self.max_features]

        if class_perm is not None:
            assert self.mode == "cls", "class_perm only makes sense for classification"
            inv_perm = np.argsort(class_perm)
            train_y = train_y.to(torch.long)
            inv_perm = torch.as_tensor(inv_perm, device=train_y.device)
            train_y = inv_perm[train_y].to(torch.float)

        return train_x, train_y, test_x, train_text, text
