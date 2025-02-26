# CHANGELOG


## v0.2.0 (2025-02-26)

### Bug Fixes

- Load specified number of rows from point cloud file
  ([#29](https://github.com/ai4trees/pointtorch/pull/29),
  [`d5c79c9`](https://github.com/ai4trees/pointtorch/commit/d5c79c932baefc394ebe11ab1dfbccdf4565d514))

- Return type of make_labels_consecutive ([#25](https://github.com/ai4trees/pointtorch/pull/25),
  [`bdfea64`](https://github.com/ai4trees/pointtorch/commit/bdfea64088cb8b8c5175b351012a53772f9a12f3))

### Continuous Integration

- Add tests for Python 3.13 and PyTorch 2.5 ([#36](https://github.com/ai4trees/pointtorch/pull/36),
  [`165e1f4`](https://github.com/ai4trees/pointtorch/commit/165e1f4c897f8c000fe896c91bbc20e55996022a))

- Install g++ in Docker image ([#46](https://github.com/ai4trees/pointtorch/pull/46),
  [`d5b9a17`](https://github.com/ai4trees/pointtorch/commit/d5b9a17bb3bf42d6b430939dda36d53c5946d1dd))

- Install g++ in Docker image ([#50](https://github.com/ai4trees/pointtorch/pull/50),
  [`86142a8`](https://github.com/ai4trees/pointtorch/commit/86142a8672d6ef841e3d48d139654507c804dbb7))

- Refactor code quality pipeline ([#35](https://github.com/ai4trees/pointtorch/pull/35),
  [`0708d27`](https://github.com/ai4trees/pointtorch/commit/0708d277d17578f53cc1722ee92991f64f3f9810))

- Update CI pipeline ([#47](https://github.com/ai4trees/pointtorch/pull/47),
  [`7c4e96e`](https://github.com/ai4trees/pointtorch/commit/7c4e96e9119c732293e5b19338bc8b001184dfe0))

- Update docs pipeline ([#49](https://github.com/ai4trees/pointtorch/pull/49),
  [`d87cd59`](https://github.com/ai4trees/pointtorch/commit/d87cd590cb58e4d495d0e24a3ea15f70ed2417ef))

### Documentation

- Fix issues in docs build ([#24](https://github.com/ai4trees/pointtorch/pull/24),
  [`b1d4e59`](https://github.com/ai4trees/pointtorch/commit/b1d4e59a6cfb6d57c6bf622ae768ec19a7f1ce5e))

- Update CI badge in readme ([#48](https://github.com/ai4trees/pointtorch/pull/48),
  [`d9325c7`](https://github.com/ai4trees/pointtorch/commit/d9325c7df2f39f2d6c2dcfff4c2cf4c74a91b464))

### Features

- File download ([#30](https://github.com/ai4trees/pointtorch/pull/30),
  [`f492736`](https://github.com/ai4trees/pointtorch/commit/f49273668fa45b0e3f31a9503870923e6fa7e4be))

- File unzipping ([#32](https://github.com/ai4trees/pointtorch/pull/32),
  [`70380b3`](https://github.com/ai4trees/pointtorch/commit/70380b3fb416ad358ea1c11c2fab9853d3aa9bad))

- Fitting of oriented bounding box ([#41](https://github.com/ai4trees/pointtorch/pull/41),
  [`b1e60f0`](https://github.com/ai4trees/pointtorch/commit/b1e60f088778d00b1e2c2fd523197400f44bbaa6))

- Load specified number of rows from point cloud file
  ([#26](https://github.com/ai4trees/pointtorch/pull/26),
  [`560b594`](https://github.com/ai4trees/pointtorch/commit/560b5947a895fb162b4bd22a1d215b0f82fb9860))

- Max pooling ([#37](https://github.com/ai4trees/pointtorch/pull/37),
  [`90160c3`](https://github.com/ai4trees/pointtorch/commit/90160c371b57a4ca82366c4055e1a4c721b88e46))

- Non-maximum suppression ([#40](https://github.com/ai4trees/pointtorch/pull/40),
  [`a3386d3`](https://github.com/ai4trees/pointtorch/commit/a3386d3cd54e2798f28e8970a17e3343700d2e70))

- Point cloud shuffling ([#39](https://github.com/ai4trees/pointtorch/pull/39),
  [`ae22c01`](https://github.com/ai4trees/pointtorch/commit/ae22c0144d252a31b4e62ea3857091cfa36a43c8))

- Random sampling ([#38](https://github.com/ai4trees/pointtorch/pull/38),
  [`5b95d86`](https://github.com/ai4trees/pointtorch/commit/5b95d867565fbf723781dc162d4903ffada1e967))

- Read and write CRS information ([#28](https://github.com/ai4trees/pointtorch/pull/28),
  [`f454bf6`](https://github.com/ai4trees/pointtorch/commit/f454bf6d5a0a74a230b5e9905f71f3f7ceafcb51))

### Refactoring

- Add pre-commit to dev dependencies ([#34](https://github.com/ai4trees/pointtorch/pull/34),
  [`f752def`](https://github.com/ai4trees/pointtorch/commit/f752deff1fee6e7bfb1337c7dc20d3cf94b7284d))

- Extend tests for pointtorch.io.download_file
  ([#33](https://github.com/ai4trees/pointtorch/pull/33),
  [`5808abd`](https://github.com/ai4trees/pointtorch/commit/5808abda1b42ba37dd1c97403bc718aa1cc8fd05))

- Fix mypy errors for pointtorch core module ([#27](https://github.com/ai4trees/pointtorch/pull/27),
  [`764ed00`](https://github.com/ai4trees/pointtorch/commit/764ed002e4da4ea30d19f40143cc5b3067a00ae6))

- Laz writer ([#42](https://github.com/ai4trees/pointtorch/pull/42),
  [`3e7f09c`](https://github.com/ai4trees/pointtorch/commit/3e7f09c2941f23c5fb37b10029854cbaaa1b96f3))

- Remove torch dependencies from pyproject.toml
  ([#43](https://github.com/ai4trees/pointtorch/pull/43),
  [`36fd411`](https://github.com/ai4trees/pointtorch/commit/36fd41116e1f128ec2e1b4026d0dc64ac1735df4))

- Remove torch dependencies from pyproject.toml
  ([#44](https://github.com/ai4trees/pointtorch/pull/44),
  [`a554893`](https://github.com/ai4trees/pointtorch/commit/a554893287c7a1f0e12395e8de705455239da28e))

- Update code formatting ([#31](https://github.com/ai4trees/pointtorch/pull/31),
  [`758a361`](https://github.com/ai4trees/pointtorch/commit/758a361f34ef1757982a367d261ae67d94b6bdd1))

- Update docs ([#45](https://github.com/ai4trees/pointtorch/pull/45),
  [`01398bc`](https://github.com/ai4trees/pointtorch/commit/01398bcff1dfacbeff0e876342316524d3af629d))


## v0.1.0 (2024-12-14)

### Continuous Integration

- Fix code coverage ([#11](https://github.com/ai4trees/pointtorch/pull/11),
  [`d64da1e`](https://github.com/ai4trees/pointtorch/commit/d64da1e2fb069f5c2dcde626cbacdea52a92cc4d))

- Update Dockerfile ([#12](https://github.com/ai4trees/pointtorch/pull/12),
  [`82a7b1c`](https://github.com/ai4trees/pointtorch/commit/82a7b1ccbcaa5d846cd522b21f7eb7ad414b5764))

### Documentation

- Update setup instructions ([#5](https://github.com/ai4trees/pointtorch/pull/5),
  [`d1b5b96`](https://github.com/ai4trees/pointtorch/commit/d1b5b96459ca3222b821f4e16f4022adc24876d8))

### Features

- Knn search ([#10](https://github.com/ai4trees/pointtorch/pull/10),
  [`a7740bb`](https://github.com/ai4trees/pointtorch/commit/a7740bb6f7dbf1ffefbd415ddd5d9098293b76a2))

- Operation to make labels consecutive ([#16](https://github.com/ai4trees/pointtorch/pull/16),
  [`bfda0f7`](https://github.com/ai4trees/pointtorch/commit/bfda0f7e4178864021d87c7d3f55419d7b81ab3b))

- Point cloud data structure ([#1](https://github.com/ai4trees/pointtorch/pull/1),
  [`dac1516`](https://github.com/ai4trees/pointtorch/commit/dac15161ee7a34d8b8a44d6a8351ae099f0ab114))

- Point cloud I/O ([#3](https://github.com/ai4trees/pointtorch/pull/3),
  [`b8c748e`](https://github.com/ai4trees/pointtorch/commit/b8c748e69a3339bb77ecb342ffc1c3291c6e5453))

- Radius search ([#19](https://github.com/ai4trees/pointtorch/pull/19),
  [`fec74f2`](https://github.com/ai4trees/pointtorch/commit/fec74f217d8d58097272747bcc56321535122b76))

- Read method for point clouds ([#6](https://github.com/ai4trees/pointtorch/pull/6),
  [`a793141`](https://github.com/ai4trees/pointtorch/commit/a793141d240b308310d785a590e2e305fb170de7))

- Voxel downsampling ([#14](https://github.com/ai4trees/pointtorch/pull/14),
  [`8daa0d7`](https://github.com/ai4trees/pointtorch/commit/8daa0d7008e61622d3efdb6ca5c9a2aef1090682))

### Refactoring

- Error message in PointCloud class ([#21](https://github.com/ai4trees/pointtorch/pull/21),
  [`f8a1d39`](https://github.com/ai4trees/pointtorch/commit/f8a1d3939ed1ced500c4cc8e7640453f2abe05d2))

- Fix pytyped marker ([#2](https://github.com/ai4trees/pointtorch/pull/2),
  [`d849582`](https://github.com/ai4trees/pointtorch/commit/d84958229e051388a82470540581585edfaa58a4))

- Fix warnings in tests ([#13](https://github.com/ai4trees/pointtorch/pull/13),
  [`03541c0`](https://github.com/ai4trees/pointtorch/commit/03541c0c7bfb0e9bf73fdd433a8001e222a39258))

- Hdf reader ([#18](https://github.com/ai4trees/pointtorch/pull/18),
  [`5174bb5`](https://github.com/ai4trees/pointtorch/commit/5174bb599a82a2b7d84401d849d3f5b81a6fa640))

- Make_labels_consecutive operation ([#23](https://github.com/ai4trees/pointtorch/pull/23),
  [`bde3a1a`](https://github.com/ai4trees/pointtorch/commit/bde3a1a48994d6af07337d5823e6e163a99819e1))

- Numpy type annotations ([#20](https://github.com/ai4trees/pointtorch/pull/20),
  [`390892f`](https://github.com/ai4trees/pointtorch/commit/390892f6793e01c6e075034f316f4e2263b320d4))

- Package imports ([#22](https://github.com/ai4trees/pointtorch/pull/22),
  [`dc60b4d`](https://github.com/ai4trees/pointtorch/commit/dc60b4d256eac89c9c4283842c4c9d9d7da2631f))

- Reading and writing of point cloud identifiers in HDF files
  ([#15](https://github.com/ai4trees/pointtorch/pull/15),
  [`0047612`](https://github.com/ai4trees/pointtorch/commit/004761261a5b26c27e028fa83e3400abb346429b))

- Reformat package imports ([#4](https://github.com/ai4trees/pointtorch/pull/4),
  [`c3b90c6`](https://github.com/ai4trees/pointtorch/commit/c3b90c67a52d2221fc5c59062bddce57e7ff54b7))

- Voxel downsampling ([#17](https://github.com/ai4trees/pointtorch/pull/17),
  [`b430358`](https://github.com/ai4trees/pointtorch/commit/b430358383f7afc738de48d66454bc6890cf2e65))

### Testing

- Add tests for PointCloud.to() method ([#8](https://github.com/ai4trees/pointtorch/pull/8),
  [`b5825cb`](https://github.com/ai4trees/pointtorch/commit/b5825cbb46b0b133f389608782934b0fd1ef3e08))

- Adds tests for pointtorch.read ([#7](https://github.com/ai4trees/pointtorch/pull/7),
  [`354a51e`](https://github.com/ai4trees/pointtorch/commit/354a51e66dc00a8f2479d16c20cb8bad7fae61ed))

- Extend tests for pointtorch.io.LasWriter ([#9](https://github.com/ai4trees/pointtorch/pull/9),
  [`7fc1429`](https://github.com/ai4trees/pointtorch/commit/7fc142920d8e900ff9051c7f99154cbafe53bc65))
