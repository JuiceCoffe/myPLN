import argparse
import os
import xml.etree.ElementTree as ET


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
LABEL_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}


def collect_ids(voc_root, year, split):
    split_path = os.path.join(voc_root, f"VOC{year}", "ImageSets", "Main", f"{split}.txt")
    with open(split_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_xml(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    entries = []
    for obj in root.findall("object"):
        difficult = int(obj.findtext("difficult", default="0"))
        if difficult:
            continue
        class_name = obj.findtext("name")
        if class_name not in LABEL_TO_ID:
            continue
        bbox = obj.find("bndbox")
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))
        entries.append((xmin, ymin, xmax, ymax, LABEL_TO_ID[class_name]))
    return width, height, entries


def build_line(image_path, entries):
    parts = [image_path]
    for xmin, ymin, xmax, ymax, class_id in entries:
        parts.extend([f"{xmin:.2f}", f"{ymin:.2f}", f"{xmax:.2f}", f"{ymax:.2f}", str(class_id)])
    return " ".join(parts)


def build_resized_line(image_path, width, height, entries, output_size):
    sx = output_size / width
    sy = output_size / height
    parts = [image_path]
    for xmin, ymin, xmax, ymax, class_id in entries:
        parts.extend([
            f"{xmin * sx:.2f}",
            f"{ymin * sy:.2f}",
            f"{xmax * sx:.2f}",
            f"{ymax * sy:.2f}",
            str(class_id),
        ])
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Prepare VOC text files for PLN training/testing.")
    parser.add_argument("--voc-root", required=True, help="Path to VOCdevkit")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/test text files")
    parser.add_argument("--size", type=int, default=448, help="Resized test size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_lines = []
    for year, split in (("2012", "train"), ("2012", "val"), ("2007", "train"), ("2007", "val")):
        for image_id in collect_ids(args.voc_root, year, split):
            image_path = os.path.join(args.voc_root, f"VOC{year}", "JPEGImages", f"{image_id}.jpg")
            annotation_path = os.path.join(args.voc_root, f"VOC{year}", "Annotations", f"{image_id}.xml")
            _, _, entries = parse_xml(annotation_path)
            if entries:
                train_lines.append(build_line(image_path, entries))

    test_lines = []
    for image_id in collect_ids(args.voc_root, "2007", "test"):
        image_path = os.path.join(args.voc_root, "VOC2007", "JPEGImages", f"{image_id}.jpg")
        annotation_path = os.path.join(args.voc_root, "VOC2007", "Annotations", f"{image_id}.xml")
        width, height, entries = parse_xml(annotation_path)
        if entries:
            test_lines.append(build_resized_line(image_path, width, height, entries, args.size))

    with open(os.path.join(args.output_dir, "train.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(train_lines))
    with open(os.path.join(args.output_dir, "test_448.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(test_lines))


if __name__ == "__main__":
    main()
