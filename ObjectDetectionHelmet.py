import streamlit as st
from ultralytics import YOLOv10
from PIL import Image, ImageOps
import os


def load_model(model_path):
    return YOLOv10(model_path)


def upload_image():
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)  # Handle image orientation
        image = image.convert("RGB")  # Convert to RGB
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        return image, uploaded_file.name
    return None, None


def save_image(image, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    image.save(path)


def detect_objects(model, image_path):
    result = model(source=image_path)[0]
    return result


def display_result(result):
    result_image = result.plot(
        show=False)  # Ensure that the plot function is displaying the correct colors
    st.image(result_image, caption='Detection Result.', use_column_width=True)
    result.save('./result.jpg')
    st.write("Saved the result as result.jpg")


def main():
    st.title('Helmet Safety Detection with YOLOv10')

    # Update with the path to your fine-tuned model if needed
    model = load_model('yolov10n.pt')

    image, image_name = upload_image()
    if image is not None:
        st.write("")
        st.write("Detecting...")

        image_path = os.path.join("images", image_name)
        save_image(image, image_path)

        result = detect_objects(model, image_path)
        display_result(result)


if __name__ == "__main__":
    main()

