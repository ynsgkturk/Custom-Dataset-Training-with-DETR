import os
import shutil


# remove irrelevant classes from dataset
irrelevant_classes = ["0", "3", "7", "8", "11"]

# Data path
IMAGE_PATH = "VisDrone2019/Train/images"
ANNOTATION_PATH = "VisDrone2019/Train/annotations"

# Data saving path
TRAIN_IMAGE_PATH = "VisDrone2019/Updated_Train/images"
TRAIN_ANNOTATION_PATH = "VisDrone2019/Updated_Train/annotations"


# We just wanna detect person and car related objects, so this function does this.
def update_data(image_path: str, annotation_path: str, image_save_path: str, annotation_save_path: str):
    """
    Assign person related classes to person,car related classes to vehicle and
    also removes other classes that marked in [irrelevant_classes].
    :param image_path: path of your images
    :param annotation_path: path of your annotation files
    :param image_save_path: path to save updated images
    :param annotation_save_path: path to save updated annotations

    """
    for img in os.listdir(image_path):
        img_path = image_path + "/" + img
        ann_path = annotation_path + "/" + img[:-3] + "txt"

        img_save = image_save_path + "/" + img
        ann_save = annotation_save_path + "/" + img[:-3] + "txt"

        lines = []
        with open(ann_path, "r") as f:
            for line in f.readlines():
                line = line.split(",")[:6]

                if line[5] not in irrelevant_classes:
                    if (line[5] == "1") or (line[5] == "2"):
                        line[5] = "1"
                    if line[5] in ["4", "5", "6", "9", "10"]:
                        line[5] = "0"

                    lines.append(line)

            with open(ann_save, "a") as sf:
                lines = [",".join(row) + "\n" for row in lines]
                sf.writelines(lines)
                sf.close()

            f.close()

        shutil.copy(img_path, img_save)

    print("finished")


def main():
    # update_data(
    #     IMAGE_PATH,
    #     ANNOTATION_PATH,
    #     TRAIN_IMAGE_PATH,
    #     TRAIN_ANNOTATION_PATH,
    # )

    update_data(
        "VisDrone2019/ValidationOld/images",
        "VisDrone2019/ValidationOld/annotations",
        "VisDrone2019/Validation/images",
        "VisDrone2019/Validation/annotations",
    )


if __name__ == "__main__":
    main()
