
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <torch/csrc/Export.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

class TORCH_API customMnistLoader : public torch::data::datasets::Dataset<customMnistLoader> {
private:
static constexpr uint32_t kTrainSize = 60000;
static constexpr uint32_t kTestSize = 10000;

static constexpr uint32_t kImageMagicNumber = 2051;
static constexpr uint32_t kTargetMagicNumber = 2049;

static constexpr uint32_t kImageRows = 28;
static constexpr uint32_t kImageColumns = 28;
static constexpr const char* kTrainImagesFilename = "train-images-idx3-ubyte";
static constexpr const char* kTrainTargetsFilename = "train-labels-idx1-ubyte";
static constexpr const char* kTestImagesFilename = "t10k-images-idx3-ubyte";
static constexpr const char* kTestTargetsFilename = "t10k-labels-idx1-ubyte";

static bool check_is_little_endian() {
  const uint32_t word = 1;
  return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

static constexpr uint32_t flip_endianness(uint32_t value) {
  return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) |
      ((value & 0xff0000u) >> 8u) | ((value & 0xff000000u) >> 24u);
}

static uint32_t read_int32(std::ifstream& stream) {
  static const bool is_little_endian = check_is_little_endian();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t value;
  AT_ASSERT(stream.read(reinterpret_cast<char*>(&value), sizeof value));
  return is_little_endian ? flip_endianness(value) : value;
}

static uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
  const auto value = read_int32(stream);
  // clang-format off
  TORCH_CHECK(value == expected,
      "Expected to read number ", expected, " but found ", value, " instead");
  // clang-format on
  return value;
}

static std::string join_paths(std::string head, const std::string& tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += tail;
  return head;
}

static torch::Tensor read_images(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  std::ifstream images(path, std::ios::binary);
  TORCH_CHECK(images, "Error opening images file at ", path);

  //const auto count = train ? kTrainSize : kTestSize;

  // From http://yann.lecun.com/exdb/mnist/
  /*expect_int32(images, kImageMagicNumber);
  expect_int32(images, count);
  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);*/

    // read magic number
  read_int32(images);
  // read count of images
  auto count = read_int32(images);

  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);
 
  auto tensor =
      torch::empty({count, 1, kImageRows, kImageColumns}, torch::kByte);
  images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());
  return tensor.to(torch::kFloat32).div_(255);
}

static torch::Tensor read_targets(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainTargetsFilename : kTestTargetsFilename);
  std::ifstream targets(path, std::ios::binary);
  TORCH_CHECK(targets, "Error opening targets file at ", path);

  //const auto count = train ? kTrainSize : kTestSize;

  //expect_int32(targets, kTargetMagicNumber);
  // read magic number
  read_int32(images);
  auto count = read_int32(images);

  auto tensor = torch::empty(count, torch::kByte);
  targets.read(reinterpret_cast<char*>(tensor.data_ptr()), count);
  return tensor.to(torch::kInt64);
}
 public:
  /// The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  /// Loads the MNIST dataset from the `root` path.
  ///
  /// The supplied `root` path should contain the *content* of the unzipped
  /// MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
  customMnistLoader(const std::string& root, Mode mode = Mode::kTrain)
    : images_(read_images(root, mode == Mode::kTrain)),
      targets_(read_targets(root, mode == Mode::kTrain)) {}

  /// Returns the `Example` at the given `index`.
torch::data::Example<> get(size_t index) {
    return {images_[index], targets_[index]};
}

  /// Returns the size of the dataset.
std::optional<size_t> size() const {
  return images_.size(0);
}

  /// Returns true if this is the training subset of MNIST.
// NOLINTNEXTLINE(bugprone-exception-escape)
bool is_train() const noexcept {
  return images_.size(0) == kTrainSize;
}

  /// Returns all images stacked into a single tensor.
const torch::Tensor& customMnistLoader::images() const {
  return images_;
}

  /// Returns all targets stacked into a single tensor.
const torch::Tensor& targets() const {
  return targets_;
}

 private:
  torch::Tensor images_, targets_;
};












