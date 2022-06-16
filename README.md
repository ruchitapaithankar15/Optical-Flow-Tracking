# Optical Flow Tracking

In this project, you'll use the Lucas Kanade algorithm to track an object from one frame to another.

# Getting Started

## Environment Setup

We'll be using Python 3 for this assignment. To test your code on tux, you'll need to run:

```
pip3 install --user imageio
```

## Skeleton Code

Skeleton code has been linked below to get you started. Though it is not necessary, you are free to add functions as you see fit to complete the assignment. You are _not_, however, allowed to import additional libraries or submit separate code files. Everything you will need has been included in the skeleton. DO NOT change how the program reads parameters or loads the files. failure to follow these rule will result in a large loss of credit.

## Example

I've included an image sequence from the [middlebury optical flow](http://vision.middlebury.edu/flow/) dataset for you to play with.

Once your homework is complete, the following example command should result in a similar output (you may be off by a few percentage points but that's ok):


```
python3 hw3.py --boundingBox 304,329,106,58  middlebury/Army/frame07.png middlebury/Army/frame14.png 
```

The output of this command was:
```
tracked object to have moved [13.859762   -0.09457114] to (317.85977, 328.90543)
```

If you're running locally or set up x-tunneling while ssh'd into tux, you can visualize the results with the `--visualize` flag:

```
python3 hw3.py --visualize --boundingBox 304,329,106,58  middlebury/Army/frame07.png middlebury/Army/frame14.png 
```

This will show the bounding box on the first image and it's tracked location on the second image:

![](visualize_result.png)




## Testing Correctness

I've provided a unit test for _some_ of the skeleton code that should help guide you in verifying the correctness of your program. To run it, execute:

```
python3 hw3_test.py -v
```

For more information about unit tests, see the [Python unittest documentation](https://docs.python.org/3/library/unittest.html).

Note that this unit test is meant to _guide_ you, not provide an exact solution. That means three very important things:
* The unit test will not be used to grade you, the final output will.
* The unit test may fail due to differences in your approach. If you're off by a little bit or a rounding error, that's OK.
* A correct unit test does not garuntee a correct solution, it just means it's correct _for that case_. You may find other inputs that make it fail.

# Submission

All of your code should be in a single file called `hw3.py`. Be sure to try to write something for each function. If your program errors on run you will loose many, if not all, points.

In addition to your code, you _must_ include the following items:

* A  `ReadMe.pdf` with the following items with a description of your experiments (e.g., changing parameters and showing their effects on the results, include lots of pictures), _some examples on images of your own_, and a short discussion of what you struggled with, if anything. If you didn't complete the assignment or there is a serious bug in your code, indicate it here. If you believe your code is awesome, just say so. If you did extra credit, discuss it in this file as well. 

Call your python file `hw3.py` and zip it, along with `ReadMe.pdf`, into an archive called `DREXELID_hw_3.zip` where `DREXELID` is your `abc123` alias. If you did extra credit and made a different executable, call it `hw3_ec.py`. DO NOT include _any_ other files

# Implementation Notes
There are some inconsistencies about how to implement the sobel kernels depending on what matrix library you use, as well as which reference you use for the equations. Specifically, the direction of the sobel-y kernel tends to flip depending on where you look it up. Typically, this doesn't matter (you can always flip the direction), but this assignment makes some assumptions about the one you use (as does the unit test). 

For the solution, I've used the sobel kernels as defined by the [wikipedia](https://en.wikipedia.org/wiki/Sobel_operator) entry. As indicated in the code, they were then negated.


# Grading
All submissions will be graded by a program run on the [tux cluster](https://www.cs.drexel.edu/Account/Account.html). The grader is similar to the provided unit test, but will use different input and outputs. It is your responsibility to ensure your submission can be run on the tux cluster (if you follow the above instructions, it will!).

To avoid runtime errors:
* do no import any libraries (this is a requirement)
* do not rely on environment variables
* do not hard code any paths into your program

The assignment is worth 50 points and will be graded as follows:
* [6 pts]: lucas_kanade function
* [1.5 pts]: iterative_lucas_kanade function
* [1.5 pts]: gaussian_pyramid function
* [3 pts]: pyramid_lucas_kanade function
* [1.5 pts]: track_object function
* [1.5 pts]: Include the report PDF described above. 

# Extra Credit Options

There are a large number of options for extra credit on this project:
* [1.5 pts]: Make the program take an arbitrary number of frames and track the object as it moves through them all.
* [3 pts]: Refactor how the pyramid is formed such that it works on the entire images. Then, make the lucas_kanade functions take the sub-images from that. Finally, use this new approach to track multiple objects by re-using the same pyramid.
* [3 pts]: Implement the Good Features To Track algorithm that rates the "goodness" of the AtA matrix. Next, use this function to find good features inside windows of a size set by the user. Finally, track the window with the highest score between the two frames (so you don't need the `--boundingBox` parameter anymore).
* [3 pts]: Make this work on video files. To achieve this, you may utilize the opencv package _only_ in `hw3_ec.py` and _only_ for the video IO and display (do not use the opencv lucas kanade functions)
* [? pts]: Do something really interesting and propose how many points you think it's worth :D.

You must show the results of your code for each extra credit portion.
