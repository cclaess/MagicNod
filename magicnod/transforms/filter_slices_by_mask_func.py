from collections.abc import Hashable, Mapping, Callable

import torch
from monai.config import KeysCollection
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import MapTransform, Randomizable


class FilterSlicesByMaskFuncd(MapTransform, Randomizable, MultiSampleTrait):
    def __init__(
        self, 
        keys: KeysCollection, 
        mask_key: str, 
        mask_filter_func: Callable[[torch.Tensor], torch.Tensor],
        slice_dim: int = 3,
        num_slices: int = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Initializes the transform with a custom mask filter function.
        
        Args:
            keys: Keys for the items to transform.
            mask_key: Key to access the mask in the input data.
            mask_filter_func: A function that takes in a mask tensor and returns
                              a boolean tensor indicating which slices to keep.
            slice_dim: Dimension along which to slice the mask and data.
            num_slices: Number of slices to keep. If None, all slices that meet the
                        mask criteria will be kept.
            allow_missing_keys: Whether to allow missing keys in data.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.mask_key = mask_key
        self.mask_filter_func = mask_filter_func
        self.slice_dim = slice_dim
        self.num_slices = num_slices

    def __call__(
        self, 
        data: Mapping[Hashable, torch.Tensor]
    ) -> list[dict[Hashable, torch.Tensor]]:
        """
        Returns a list of dictionaries containing the slices that meet the custom
        mask criteria as defined by `mask_filter_func`.
        
        Args:
            data: A dictionary containing the data, including the mask tensor.

        Returns:
            A list of dictionaries where each dictionary contains the data for a
            slice that meets the mask criteria.
        """
        mask = data[self.mask_key]

        # Check if `slice_dim` is valid
        if self.slice_dim < 0 or self.slice_dim >= mask.dim():
            raise ValueError(f"Invalid `slice_dim` value: {self.slice_dim}")
        
        # Apply the custom filter function and check its output shape
        mask_criteria = self.mask_filter_func(mask)
        expected_shape = mask.shape[self.slice_dim]

        if not isinstance(mask_criteria, torch.Tensor):
            raise TypeError(
                f"mask_filter_func must return a tensor, but got {type(mask_criteria)}."
            )
        
        if mask_criteria.shape != (expected_shape,):
            raise ValueError(
                f"mask_filter_func must return a tensor of shape {expected_shape}, "
                f"but got {mask_criteria.shape}."
            )
        
        # Get the indices of the slices that meet the criteria
        valid_slices = [i for i in range(expected_shape) if mask_criteria[i].item()]

        # Randomly select a subset of slices if `num_slices` is specified
        if self.num_slices is not None and len(valid_slices) > self.num_slices:
            valid_slices = self.R.choice(valid_slices, self.num_slices, replace=False)

        data_list = []
        for i in valid_slices:
            data_list.append({
                key: data[key].select(self.slice_dim, i) for key in self.keys
            })
        return data_list


if __name__ == "__main__":

    # Example usage
    data = {
        "image": torch.rand(1, 256, 256, 128),
        "mask": torch.zeros(1, 256, 256, 128),
    }
    data["mask"][0, 100:200, 100:200, 50:70] = 1
    
    filter_slices = FilterSlicesByMaskFuncd(
        keys=["image", "mask"],
        mask_key="mask",
        mask_filter_func=lambda mask: mask.sum(dim=(0, 1, 2)) == 0,  # Keep slices with no mask
        slice_dim=3,
        num_slices=25,
    )

    filtered_data = filter_slices(data)

    print(f"Original data shape: {data['image'].shape}")
    print(f"Filtered data shape: {filtered_data[0]['image'].shape}")
    print(f"Number of slices: {len(filtered_data)}")
