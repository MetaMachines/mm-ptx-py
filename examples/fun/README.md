# MM-PTX Fun Examples

[domain_coloring](https://en.wikipedia.org/wiki/Domain_coloring)

## Domain Coloring
This example uses a PTX Inject kernel that takes two functions meant to be derivatives and plots them by their vector angle and 
magnitude using the HSL colorspace. An example instruction set is provided in Stack PTX. The
instructions are compiled into a cubin and run to create the png stored in the folder. Running the example also
creates a `mp4` video of the plot being animated.

![domain coloring example](domain_coloring/domain_coloring_output.png)

## Domain Coloring Random
This example also uses domain coloring to generating HSL images of functions but instead the functions are generated in `generator_instructions.py`. 64 sets of Stack PTX instructions are generated and each in turn is compiled and ran. The 
output gifs are montaged to assemble the image below:

![domain coloring random example](domain_coloring_random/domain_coloring_output.gif)
