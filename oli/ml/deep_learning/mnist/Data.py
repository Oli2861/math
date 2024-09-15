from oli.math.math_utility import pretty_print_matrix


class Image:
    pixels: list[list[int]]
    height: int
    width: int

    def __init__(self, pixels: list[list[int]]):
        self.pixels = pixels
        self.height = len(pixels)
        self.width = len(pixels[0])

    def print(self):
        pretty_print_matrix(self.pixels, label=f"Image with dimensions width = {self.width} x height = {self.height}",
                            max_length=3)

    def get_linearized(self) -> list[int]:
        res: list[int] = []
        for row in self.pixels:
            for item in row:
                res.append(item)
        return res


class MNISTDataset:
    images: list[Image]
    labels: list[int]

    def __init__(self, images: list[Image], labels: list[int]):
        self.images = images
        self.labels = labels

        if len(images) != len(labels):
            raise Exception("Amount of images doesnt match amount of labels.")

    def __iter__(self):
        return ((self.images[i], self.labels[i]) for i in range(len(self.images)))

    def get_linearized_images(self) -> list[list[int]]:
        return [img.get_linearized() for img in self.images]


def read_image_file(path: str) -> list[Image]:
    file_stream = open(path, "rb")
    # Offset 0 - 4 --> 4 bytes
    magic_number: bytes = file_stream.read(4)
    magic_number: int = int.from_bytes(magic_number, byteorder="big", signed=False)

    # Offset 4 - 8 --> 4 bytes
    number_of_images: bytes = file_stream.read(4)
    number_of_images: int = int.from_bytes(number_of_images, byteorder="big", signed=False)

    # Offset 8 - 12 --> 4 bytes
    number_of_rows: bytes = file_stream.read(4)
    number_of_rows: int = int.from_bytes(number_of_rows, byteorder="big", signed=False)

    # Offset 8 - 12 --> 4 bytes
    number_of_columns: bytes = file_stream.read(4)
    number_of_columns: int = int.from_bytes(number_of_columns, byteorder="big", signed=False)

    print(
        f"Loading images:\tMagic number: {magic_number}, Number of images: {number_of_images}, Number of rows: {number_of_rows}, Number of columns: {number_of_columns}")

    images: list[Image] = []
    count = 0
    for image_number in range(number_of_images):
        pixels: list[list[int]] = [[0 for n in range(number_of_columns)] for i in range(number_of_rows)]
        for row_number in range(number_of_rows):
            for column_number in range(number_of_columns):
                pixel: bytes = file_stream.read(1)
                pixel: int = int.from_bytes(pixel, byteorder="big", signed=False)
                pixels[row_number][column_number] = pixel
        images.append(Image(pixels))
        if image_number % 10000 == 0:
            print("Loaded image number", image_number)
    return images


def read_label_file(path: str):
    file_stream = open(path, "rb")
    # Offset 0 - 4 --> 4 bytes
    magic_number: bytes = file_stream.read(4)
    magic_number: int = int.from_bytes(magic_number, byteorder="big", signed=False)

    # Offset 4 - 8 --> 4 bytes
    number_of_items: bytes = file_stream.read(4)
    number_of_items: int = int.from_bytes(number_of_items, byteorder="big", signed=False)

    print(f"Loading labels:\tMagic number: {magic_number}, Number of items: {number_of_items}")

    items: list[int] = []
    for item_index in range(number_of_items):
        item: bytes = file_stream.read(1)
        item: int = int.from_bytes(item, byteorder="big", signed=False)
        items.append(item)

    return items
