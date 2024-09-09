# computer-vision

An app to connect computer vision and emotions

Current progress:<br>
Emotions training set - 70% accuracy<br>
Emotions testing set - ~53% accuracy

To train/run your own model:
1. Find the model in src/emo_classifier.py
2. Make any adjustments as necessary
3. Run emo_classifier.py from the command line, the model will be saved to src/res/models
4. Additionally, add the -t or --test flag to disable training to run the model set in the Config class
