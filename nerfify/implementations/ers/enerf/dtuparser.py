# @dataclass
# class DataparserOutputs:
#     """Dataparser outputs for the which will be used by the DataManager
#     for creating RayBundle and RayGT objects."""

#     image_filenames: List[Path]
#     """Filenames for the images."""
#     cameras: Cameras
#     """Camera object storing collection of camera information in dataset."""
#     alpha_color: Optional[TensorType[3]] = None
#     """Color of dataset background."""
#     scene_box: SceneBox = SceneBox()
#     """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
#     mask_filenames: Optional[List[Path]] = None
#     """Filenames for any masks that are required"""
#     metadata: Dict[str, Any] = to_immutable_dict({})
#     """Dictionary of any metadata that be required for the given experiment.
#     Will be processed by the InputDataset to create any additional tensors that may be required.
#     """
#     dataparser_transform: TensorType[3, 4] = torch.eye(4)[:3, :]
#     """Transform applied by the dataparser."""
#     dataparser_scale: float = 1.0
#     """Scale applied by the dataparser."""

# @dataclass
# class DataParser:

#     @abstractmethod
#     def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
#         """Abstract method that returns the dataparser outputs for the given split.

#         Args:
#             split: Which dataset split to generate (train/test).

#         Returns:
#             DataparserOutputs containing data for the specified dataset and split
#         """