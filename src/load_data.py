import tensorflow as tf

data_dir = r"C:\Users\Prashant\.keras\datasets\flower_photos\flower_photos"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224,224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224,224),
    batch_size=32
)

print("Classes:", train_ds.class_names)

for images, labels in train_ds.take(1):
    print("Image Batch Shape:", images.shape)
    print("Label Shape:", labels.shape)