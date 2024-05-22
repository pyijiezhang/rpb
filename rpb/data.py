import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def loaddataset(name):
    """
    Loads the specified dataset (MNIST or FashionMNIST) with predefined transformations.

    Args:
        name (str): The name of the dataset to load. Should be either 'mnist' or 'fmnist'.

    Returns:
        tuple: A tuple containing the training and test datasets.

    Raises:
        RuntimeError: If the provided dataset name is not 'mnist' or 'fmnist'.

    Example:
        >>> train, test = loaddataset('mnist')
    """

    # Set the random seed for reproducibility
    torch.manual_seed(7)

    if name == "mnist":
        # Define the transformation for MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        # Load the MNIST training and test datasets
        train = datasets.MNIST(
            "mnist-data/", train=True, download=True, transform=transform
        )
        test = datasets.MNIST(
            "mnist-data/", train=False, download=True, transform=transform
        )
    elif name == "fmnist":
        # Define the transformation for FashionMNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.286,), (0.353,))]
        )
        # Load the FashionMNIST training and test datasets
        train = datasets.FashionMNIST(
            "fmnist-data/", train=True, download=True, transform=transform
        )
        test = datasets.FashionMNIST(
            "fmnist-data/", train=False, download=True, transform=transform
        )
    else:
        # Raise an error if the dataset name is incorrect
        raise RuntimeError(f"Wrong dataset chosen {name}")

    return train, test


def loadbatches_train(train, loader_kargs, batch_size, T_splits, seed):
    """
    Creates multiple DataLoader instances for training data, split according to specified proportions.

    Args:
        train (Dataset): The training dataset.
        loader_kargs (dict): Additional keyword arguments for the DataLoader.
        batch_size (int): The batch size for each DataLoader.
        T_splits (list): A list of integers specifying the size of each split.
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of DataLoader instances for each split of the training data.

    Example:
        >>> train_loaders = loadbatches_train(train_dataset, {'num_workers': 2}, 64, [10000, 20000, 30000], 42)
    """

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Get the number of samples in the training dataset
    n_train = len(train.data)

    # Generate a list of indices and shuffle them
    indices = list(range(n_train))
    np.random.shuffle(indices)

    # Calculate the number of splits
    T = len(T_splits)

    # Initialize a list to hold the DataLoader instances
    train_loaders = []

    # Create a DataLoader for each split
    for t in range(T):
        # Determine the indices for the current split
        subset_indices = indices[sum(T_splits[:t]) : sum(T_splits[: t + 1])]

        # Create a SubsetRandomSampler using the subset indices
        train_sampler = SubsetRandomSampler(subset_indices)

        # Create a DataLoader for the current split
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=train_sampler, **loader_kargs
        )

        # Add the DataLoader to the list
        train_loaders.append(train_loader)

    return train_loaders


def loadbatches_eval(train, loader_kargs, batch_size, T_splits, seed):
    """
    Creates multiple DataLoader instances for evaluation data, split according to specified proportions.

    Args:
        train (Dataset): The dataset for evaluation.
        loader_kargs (dict): Additional keyword arguments for the DataLoader.
        batch_size (int): The batch size for each DataLoader.
        T_splits (list): A list of integers specifying the size of each split.
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of DataLoader instances for each split of the evaluation data.

    Example:
        >>> eval_loaders = loadbatches_eval(train_dataset, {'num_workers': 2}, 64, [10000, 20000, 30000], 42)
    """

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Get the number of samples in the dataset
    n_eval = len(train.data)

    # Generate a list of indices and shuffle them
    indices = list(range(n_eval))
    np.random.shuffle(indices)

    # Calculate the number of splits
    T = len(T_splits)

    # Initialize a list to hold the DataLoader instances
    eval_loaders = []

    # Create a DataLoader for each split
    for t in range(T):
        if t == 0:
            # Use all indices for the first split
            subset_indices = indices[:]
        else:
            # Use indices from the current split to the end
            subset_indices = indices[sum(T_splits[:t]) :]

        # Create a SubsetRandomSampler using the subset indices
        eval_sampler = SubsetRandomSampler(subset_indices)

        # Create a DataLoader for the current split
        eval_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=eval_sampler, **loader_kargs
        )

        # Add the DataLoader to the list
        eval_loaders.append(eval_loader)

    return eval_loaders
