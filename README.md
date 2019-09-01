## Image classification app

This image classification app was built based on the tutorial [TensorFlow.js Transfer Learning Image Classifier](https://codelabs.developers.google.com/codelabs/tensorflowjs-teachablemachine-codelab/index.html#0).
Every task from the tutorial is made using different branches and then merged into master. 

### Layout
Layout for this app is an webcam screen together with 4 buttons. Each button represents a class. Every time a user clicks on a button, a webcam image is added to a particular class as an training example [[tutorial-task-7]](https://codelabs.developers.google.com/codelabs/tensorflowjs-teachablemachine-codelab/index.html#6).

![Layout - webcam and 4 buttons](https://github.com/markat1/mt-image-classifier-assignment/blob/master/images/layout.jpg)

### Method – how I tested the app!

How I tested this app is showing an item (like a deodorant) in front of the webcam. I click a bunch of times on the button “Add A”.

 Then I show a different item for the webcam and I clicked a bunch of times on button “add B”. Same method goes for class C and “No action”.  

I chose representing class “no action” as the wall in the background. That means that when there’s no items shown for the webcam,  then it should predict “no action”. 

I tried holding different items (1 item at a time) over the webcam. I was testing if it was able to predict what class the item belongs to and probability of how likely it belongs to that specific class. 

![Classes - A, B, C and "No action"](https://github.com/markat1/mt-image-classifier-assignment/blob/master/images/classes.jpg)


### Problems and how I improved the model

As long as I hold the item in almost the same spot as when I made the test images for the item, then it has no problem in predicting with high probability what category/class the item belongs to. 

If I just changed the camera angle, the model had problems figuring out the correct class or only predict the right class with a low predictability. 

When the model had problems predicting the right item category,I just added a bunch more test images for that particular items class. I showed the webcam the particular item and then click a bunch of times on that particular button/class it should belong to. I made test images from multiple angles. 

Afterwards I tested again by showing the webcam the item. What I saw was an improvement in predicting the right class and with a higher probability. 

### Transfer learning
In task 7 we stop using mobileNets label and probability. Instead the tutorial wants MobileNet to stop right after it  makes the activation map from the webcam image via covnet layers. The activation map is used instead as input for the K nearest neighbours classifier. K nearest neighbours classifier predicts by comparing our activation map with the test activations maps, that we made using my the A,B,C and “no action” buttons in the DOM. It will try find the most similar test activation with the one we are making predictions for. It will output the result in the browser.
[[tutorial-task-7]](https://codelabs.developers.google.com/codelabs/tensorflowjs-teachablemachine-codelab/index.html#6)
[[the-coding-train]](https://youtu.be/kRpZ5OqUY6Y?t=365)

![Transfer learning - Using MobileNet as pre model to K-nearest neighbours"](https://github.com/markat1/mt-image-classifier-assignment/blob/master/images/transfer_learning.jpg)

Using Mobilenet as a pre model to KNN called transfer learning. Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. [[transfer-learning-mastery]](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/) [[transfer-learning-siraj]](https://www.youtube.com/watch?v=Ui1KbmutX0k) 

Deep convolutional neural network can take days or event weeks to train on very large dataset. Then using pre-trained models will short-cut this process. Pre model works  great with small datasets as well, because it’s weight already been trained on a lot of images. [[transfer-learning-cs231n]](http://cs231n.github.io/transfer-learning/)

### Image classification and convolutional neural network
Convolutional neural network is a deep neural network most often seen used with image classification. Image classifcation is a way of taking an image and ouput an class that best describe the image. 

images is seen as pixels. CNN tries via filters/feature identfiers to find features on image like edges and curves and tries building something out of these concepts. It does that via convolving that is like shining an image with a flashlight.

After convolving what we have is an activation map.

[[CCN adeshpande3
]](https://adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/) [[CCN siraj
]](https://www.youtube.com/watch?v=FTr3n7uBIuE)




### How the important parts of the code works
The code that I will focus on is split up in 2 parts: 
  - Data collection 
  - Prediction. 
  
#### Data collection
If we want to add more test images to our class we click one of the class buttons in the DOM. When that happens an event i fired and calls a callback function called addExample. addExample takes and index and a class. 

 ````
 const addExample = classId => {
        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(webcamElement, 'conv_preds');

        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);
    };

    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));
    document.getElementById('class-no-action').addEventListener('click', () => addExample(3));    
````
This code snippet can be seen via this link : [Code](https://github.com/markat1/mt-image-classifier-assignment/blob/master/index.js#L16-L29)


Inside the addExample function Mobilenet infer functions is called with webcam image as  an argument and a string indicating that we only want the Mobilenet to process our image with the convolutional layers. This function will output an activation map that is used as input for the K-Nearest Neighbors Classifier.

#### Prediction

The prediction happens in the endless while loop. It first checks if we have already added some activations maps and classes to our KNN classifier, and if that is the case then it calls mobileNets infer function with the webcame image as input. It specifies that we only want to get the activation map. We use the activation map to predict what class it likely is connected to the most using K nearest neightbour. It compares our activation map with our test activation maps that are most similar to our activation map. Prediction is then showed in DOM together with its probability.   

````
 while (true) {
        if (classifier.getNumClasses() > 0) {
            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(webcamElement, 'conv_preds');
            // Get the most likely class and confidences from the classifier module.
            const result = await classifier.predictClass(activation);

            const classes = ['A', 'B', 'C', 'No action'];
            document.getElementById('console').innerText = `
          prediction: ${classes[result.classIndex]}\n
          probability: ${result.confidences[result.classIndex]}
        `;
        }

        await tf.nextFrame();
    }
````
This code snippet can be seen via this link : [Code](https://github.com/markat1/mt-image-classifier-assignment/blob/master/index.js#L31-L47)

