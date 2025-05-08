import torch

def test_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("Current Device Index:", torch.cuda.current_device())

        # Test a simple tensor operation on GPU
        x = torch.tensor([1.0, 2.0], device='cuda')
        y = torch.tensor([3.0, 4.0], device='cuda')
        z = x + y
        print("Tensor result (on GPU):", z)
    else:
        print("CUDA is not available. Make sure you installed PyTorch with CUDA and have a compatible GPU.")

if __name__ == "__main__":
    test_cuda()
