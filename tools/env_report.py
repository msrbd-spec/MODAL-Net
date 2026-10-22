"""Environment and hardware diagnostics for MODAL-Net.

Prints Python, PyTorch, CUDA, cuDNN, timm, albumentations, and related dependency
versions, checks hardware acceleration (CUDA/MPS), and verifies whether timm
can instantiate required model backbone architectures without downloading.
"""

import sys
import platform
import yaml
import torch


def main():
    print("=" * 70)
    print("MODAL-Net Environment Report")
    print("=" * 70)

    # 1. System and Python
    print(f"OS:                {platform.platform()} ({platform.machine()})")
    print(f"Python:            {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Python Executable: {sys.executable}")

    # 2. PyTorch and Acceleration
    print(f"PyTorch:           {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:    {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Version:      {torch.version.cuda}")
        print(f"Device Count:      {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}:          {torch.cuda.get_device_name(i)}")
        cudnn_ver = torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') else 'N/A'
        print(f"cuDNN Version:     {cudnn_ver}")
    else:
        print("CUDA Version:      None (no CUDA GPU in this test environment)")
        mps_avail = getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
        print(f"Apple MPS:         {mps_avail}")
        print("cuDNN Version:     UNTESTED (no CUDA)")

    # 3. Vision & Scientific Packages
    packages = [
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
        ('albumentations', 'albumentations'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('yaml', 'pyyaml'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'),
        ('pytest', 'pytest'),
        ('PIL', 'Pillow'),
    ]

    print("\nDirect Dependencies:")
    for mod_name, disp_name in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, '__version__', 'unknown')
            print(f"  {disp_name:<20}: {ver}")
        except ImportError:
            print(f"  {disp_name:<20}: NOT INSTALLED")

    # 4. Timm Backbone Creation (offline verification with pretrained=False)
    import timm
    backbone_ids = [
        'maxvit_large_tf_512.in21k_ft_in1k',
        'convnextv2_large.fcmae_ft_in22k_in1k',
        'tf_efficientnetv2_m.in21k_ft_in1k',
    ]

    print("\nTimm Backbone Instantiation (pretrained=False, offline mode):")
    for b_id in backbone_ids:
        try:
            m = timm.create_model(b_id, pretrained=False, num_classes=0)
            n_params = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"  [OK] {b_id:<38} ({n_params:.1f}M params)")
        except Exception as e:
            print(f"  [FAIL] {b_id:<36}: {e}")

    print("=" * 70)


if __name__ == '__main__':
    main()
