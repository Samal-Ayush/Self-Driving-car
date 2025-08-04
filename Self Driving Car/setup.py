from setuptools import setup, find_packages

setup(
    name="selfdriving_fsd",
    version="0.1.0",
    description="End-to-end self-driving car simulation and inference pipeline with steering angle prediction and segmentation.",
    author="Ayush Samal",
    author_email="ayushsamal2003@gmail.com.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires='>=3.8',
    install_requires=[
        "tensorflow==1.15.0",         # Use your compatible tf version (e.g., 1.15 or via tensorflow-cpu)
        "opencv-python>=4.0.0",
        "numpy>=1.19.0",
        "ultralytics>=8.0.0",         # Make sure to set a compatible version for YOLO if needed
        "types-tqdm",                 # For typing hints if desired (optional)
    ],
    extras_require={
        "dev": ["pytest", "mypy", "black"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "run_steering_angle_pred=run_steering_angle_pred:main",
            "run_segmentation_obj=run_segmentation_obj:main",
            "run_fsd_inference=run_fsd_inference:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
