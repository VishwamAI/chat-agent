import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from transformers import GPT2Tokenizer, TFGPT2Model
from PIL import Image
# from glide_text2im.download import load_checkpoint
# from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults, model_and_diffusion_defaults_upsampler
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VishwamAI:
    def __init__(self, batch_size):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan(self.generator, self.discriminator)
        self.nlp_model, self.tokenizer = self.build_nlp_model()
        self.sample_dataset = self.load_sample_dataset(batch_size)

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(100,)))
        model.add(layers.Dense(135 * 135 * 16, activation='tanh'))
        model.add(layers.Reshape((135, 135, 16)))
        model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(1, 1), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(32, (4, 4), strides=(1, 1), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(3, (4, 4), strides=(1, 1), padding='same', activation='tanh'))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def build_discriminator(self):
        """
        Builds the discriminator model for the GAN.

        Returns:
            tensorflow.keras.Model: The discriminator model.
        """
        model = models.Sequential()
        model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=(1080, 1080, 3)))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_gan(self, generator, discriminator):
        """
        Builds the GAN by combining the generator and discriminator models.

        Args:
            generator (tensorflow.keras.Model): The generator model.
            discriminator (tensorflow.keras.Model): The discriminator model.

        Returns:
            tensorflow.keras.Model: The combined GAN model.
        """
        discriminator.trainable = False
        gan_input = layers.Input(shape=(100,))
        generated_image = generator(gan_input)
        gan_output = discriminator(generated_image)
        gan = models.Model(gan_input, gan_output)
        gan.compile(optimizer='adam', loss='binary_crossentropy')
        return gan

    def build_nlp_model(self):
        """
        Builds the NLP model using the GPT-2 architecture.

        Returns:
            tuple: A tuple containing the NLP model and tokenizer.
        """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = TFGPT2Model.from_pretrained("gpt2")
        dummy_input = tf.constant([[0] * 10])  # Dummy input with shape (1, 10)
        model(dummy_input)  # Build the model with the dummy input
        return model, tokenizer

    def load_sample_dataset(self, batch_size):
        """
        Loads and preprocesses the sample dataset for training.

        Args:
            batch_size (int): The batch size for training.

        Returns:
            tensorflow.data.Dataset: The preprocessed sample dataset.
        """
        def preprocess_image(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [1080, 1080])
            image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
            return image

        # Verify and adjust the path to the sample dataset if necessary
        sample_dataset_path = '/home/ubuntu/VishwamAI/data/sample_dataset'
        if not os.path.exists(sample_dataset_path):
            logging.error(f"Sample dataset path does not exist: {sample_dataset_path}")
            return None

        image_paths = [os.path.join(sample_dataset_path, image_path)
                       for image_path in os.listdir(sample_dataset_path)
                       if image_path.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_paths:
            logging.error(f"No images found in the sample dataset path: {sample_dataset_path}")
            return None

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def train(self, epochs, batch_size):
        """
        Implements the training loop for the Generative Adversarial Network (GAN).

        Args:
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size for training.

        Returns:
            None
        """
        half_batch = int(batch_size / 2)
        logging.info(f"Batch size: {batch_size}, Half batch: {half_batch}")
        for epoch in range(epochs):
            try:
                self.train_step(half_batch, batch_size, epoch)
                if epoch % 1000 == 0:
                    self.save_models(epoch)
            except Exception as e:
                logging.error(f"Error during training at epoch {epoch}: {e}")

    def train_step(self, half_batch, batch_size, epoch):
        for real_images in self.sample_dataset:
            noise = np.random.normal(0, 1, (half_batch, 100))
            generated_images = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(real_images[:half_batch], np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(generated_images, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            noise = np.random.normal(0, 1, (batch_size, 100))
            valid_y = np.array([1] * batch_size)
            g_loss = self.gan.train_on_batch(noise, valid_y)
            logging.info(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]}] [G loss: {g_loss}]")

    def save_models(self, epoch):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.generator.save(f'models/generator_epoch_{epoch}.h5')
        self.discriminator.save(f'models/discriminator_epoch_{epoch}.h5')

    def generate_question(self, input_text):
        """
        Generates a question based on input text using the NLP model.

        Args:
            input_text (str): The input text for generating the question.

        Returns:
            str: The generated question.
        """
        try:
            question_starters = [
                "What", "Why", "How", "When", "Where", "Who", "Which", "Whom", "Whose", "Can", "Could", "Would", "Should", "Is", "Are", "Was", "Were", "Will", "Shall", "Do", "Does", "Did", "Has", "Have", "Had"
            ]
            starter = np.random.choice(question_starters)
            prompt = f"{starter} {input_text}"

            logging.info("Encoding input text for question generation.")
            tokens = self.tokenizer.encode(prompt, return_tensors='tf', dtype=tf.int32)
            logging.info("Generating question from tokens.")
            outputs = self.nlp_model.generate(tokens, max_length=50, num_return_sequences=1)
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info("Question generation successful.")
            return question
        except Exception as e:
            logging.error(f"Error during question generation: {e}")
            return None

    def self_improve(self):
        """
        Orchestrates the self-improvement process of the VishwamAI model.

        Returns:
            None
        """
        performance_metrics = self.evaluate_performance()
        logging.info(f"Performance metrics: {performance_metrics}")
        new_data = self.search_new_data()
        logging.info(f"New data found: {len(new_data)} images")
        self.integrate_new_data(new_data)
        logging.info("New data integrated into the training process")
        self.train(epochs=1000, batch_size=32)
        logging.info("Model training updated with new data")

    def evaluate_performance(self):
        """
        Evaluates the performance of the model using FID and IS metrics.

        The Frechet Inception Distance (FID) and Inception Score (IS) are used to evaluate the quality of generated images.
        - FID measures the distance between the feature vectors of real and generated images, with lower values indicating better quality.
        - IS measures the diversity and quality of generated images, with higher values indicating better quality.

        Returns:
            dict: A dictionary containing the FID and IS scores.
        """
        try:
            logging.info("Loading InceptionV3 model for performance evaluation.")
            model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

            logging.info("Generating images for evaluation.")
            noise = np.random.normal(0, 1, (100, 100))
            generated_images = self.generator.predict(noise)
            generated_images = tf.image.resize(generated_images, (299, 299))
            generated_images = preprocess_input(generated_images)

            logging.info("Calculating activations for generated images.")
            act_gen = model.predict(generated_images)
            mu_gen = np.mean(act_gen, axis=0)
            sigma_gen = np.cov(act_gen, rowvar=False)

            logging.info("Selecting real images for evaluation.")
            idx = np.random.randint(0, self.sample_dataset.shape[0], 100)
            real_images = self.sample_dataset[idx]
            real_images = tf.image.resize(real_images, (299, 299))
            real_images = preprocess_input(real_images)

            logging.info("Calculating activations for real images.")
            act_real = model.predict(real_images)
            mu_real = np.mean(act_real, axis=0)
            sigma_real = np.cov(act_real, rowvar=False)

            logging.info("Calculating Frechet Inception Distance (FID).")
            ssdiff = np.sum((mu_gen - mu_real)**2.0)
            covmean = sqrtm(sigma_gen.dot(sigma_real))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = ssdiff + np.trace(sigma_gen + sigma_real - 2.0 * covmean)

            logging.info("Calculating Inception Score (IS).")
            p_yx = act_gen
            p_y = np.expand_dims(np.mean(p_yx, axis=0), 0)
            kl_d = p_yx * (np.log(p_yx + 1e-10) - np.log(p_y + 1e-10))
            is_score = np.exp(np.mean(np.sum(kl_d, axis=1)))

            logging.info(f"Performance evaluation completed. FID: {fid}, IS: {is_score}")
            return {'FID': fid, 'IS': is_score}
        except Exception as e:
            logging.error(f"Error evaluating performance: {e}")
            return {'FID': None, 'IS': None}

    def search_new_data(self):
        """
        Scrapes images from a specified URL and saves them to a local directory.

        Returns:
            list: A list of file paths to the downloaded images.
        """
        import requests
        from bs4 import BeautifulSoup
        import urllib

        # Define the URL to scrape images from
        url = "https://unsplash.com/s/photos/sample"

        try:
            # Send a request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all image tags
            img_tags = soup.find_all("img")

            # Create a directory to save the new images
            new_data_dir = "data/new_data"
            if not os.path.exists(new_data_dir):
                os.makedirs(new_data_dir)

            # Download and save the images
            new_data = []
            for img in img_tags:
                img_url = img.get("src")
                if img_url and img_url.startswith("http"):
                    try:
                        img_data = requests.get(img_url).content
                        img_name = os.path.join(new_data_dir, os.path.basename(urllib.parse.urlparse(img_url).path))
                        with open(img_name, "wb") as handler:
                            handler.write(img_data)
                        new_data.append(img_name)
                    except Exception as e:
                        logging.error(f"Error downloading image {img_url}: {e}")

            logging.info(f"Successfully downloaded {len(new_data)} images from {url}")
            return new_data

        except requests.exceptions.RequestException as e:
            logging.error(f"Error accessing {url}: {e}")
            return []

    def integrate_new_data(self, new_data):
        """
        Loads and preprocesses new image data, normalizing it and appending it to the sample dataset.

        Args:
            new_data (list): A list of file paths to the new images to be integrated.
        """
        def preprocess_image(image_path):
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(1080, 1080))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
            return image

        # Load and preprocess the new data
        new_images = []
        for image_path in new_data:
            if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    new_images.append(preprocess_image(image_path))
                except Exception as e:
                    logging.error(f"Error loading image {image_path}: {e}")
            else:
                logging.warning(f"Unsupported file format: {image_path}")

        if new_images:
            new_dataset = tf.data.Dataset.from_tensor_slices(new_images)
            self.sample_dataset = self.sample_dataset.concatenate(new_dataset)

    def generate_image(self, input_text, target_resolution=(512, 512)):
        """
        Generates an image based on input text using the NLP model and generator.

        Args:
            input_text (str): The input text for generating the image.
            target_resolution (tuple): The desired resolution of the generated image (width, height).

        Returns:
            numpy.ndarray: The generated image as a NumPy array.
        """
        try:
            logging.info("Starting image generation process.")

            # Process the input text using the NLP model
            logging.info("Encoding input text.")
            tokens = self.tokenizer.encode(input_text, return_tensors='tf')
            logging.info("Generating NLP output.")
            nlp_output = self.nlp_model(tokens)[0]

            # Generate noise vector based on NLP output
            logging.info("Generating noise vector.")
            noise = np.random.normal(0, 1, (1, 100))
            nlp_output = nlp_output.numpy().flatten()
            noise[0, :min(100, len(nlp_output))] = nlp_output[:min(100, len(nlp_output))]

            # Generate the image using the generator model at a lower resolution
            logging.info("Generating image using the generator model.")
            low_res_image = self.generator.predict(noise)

            # Resize the generated image to the target resolution
            logging.info(f"Resizing image to target resolution: {target_resolution}.")
            generated_image = tf.image.resize(low_res_image, target_resolution).numpy()

            logging.info("Image generation successful.")
            return generated_image
        except Exception as e:
            logging.error(f"Error during image generation: {e}")
            return None

def test_data_generator(batch_size=2):
    vishwamai = VishwamAI(batch_size=batch_size)
    dataset = vishwamai.load_sample_dataset(batch_size)
    for i, batch in enumerate(dataset.take(5)):
        print(f"Batch {i+1}: {batch.shape}")
    print("Data generator test completed successfully.")

def test_logging():
    logging.info("This is an info message for testing.")
    logging.warning("This is a warning message for testing.")
    logging.error("This is an error message for testing.")
    logging.critical("This is a critical message for testing.")
    print("Logging test completed successfully.")

if __name__ == "__main__":
    test_data_generator(batch_size=2)

def test_generate_images(input_text, num_images=10, output_dir="generated_images"):
    vishwamai = VishwamAI(batch_size=32)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(num_images):
        generated_image = vishwamai.generate_image(input_text)
        generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)  # Denormalize to [0, 255]
        tf.keras.preprocessing.image.save_img(f"{output_dir}/generated_image_{i}.png", generated_image[0])
    print(f"Generated {num_images} images based on the input text: '{input_text}'")
