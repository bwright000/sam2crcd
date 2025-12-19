# Initial Prompting Test Results
Inital prompting was done through 4 separate tests on a spread of videos - 5 data splits were chosen at random - for each datasplit 4 tests were ran on the initial frame, to create 4 sets of masks, each initial mask was then propagated over the remainder of the video using SAM2. These results were then compared with the Ground Truth masks provided in the CRCD dataset, with a DICE loss computed for each frame, and in turn an average determined for the full video
## Prompt Parameters
As previously stated, 4 separate tests were carried out - These are shown below:

### Test 1 - No Prompt
The 'no prompt' provided SAM2 with a single bounding box covering the entire image - with the effective instruction of 'Find the most relevant object in this frame'

### Test 2 - 3 Click Prompting
3 Click prompting provided 3 positive clicks to each 'object' in an image - each positive click was provided on the relevant object associated.

### Test 3 - Bounding Box  Prompting
Bounding Box prompting provided a single bounding box (With no positive or negative clicks) surrounding each object. With the effective instruction of 'identify the most prominant object in this frame'

### Test 4 - Combined Prompting
Combined prompting utilised both Test 2 and Test 3

## Comparison to the CRCD Dataset


