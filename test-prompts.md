# Initial Prompting Test Results
Inital prompting was done through 4 separate tests on a spread of videos - 5 data splits were chosen at random - for each datasplit 4 tests were ran on the initial frame, to create 4 sets of masks, each initial mask was then propagated over the remainder of the video using SAM2. An example slide of the prompter is shown below:
<img width="2470" height="1540" alt="image" src="https://github.com/user-attachments/assets/88fc652a-daef-4313-b0ce-e7d2e453edc6" />

## Prompt Parameters
As previously stated, 4 separate tests were carried out - These are shown below:

### Test 1 - No Prompt
The 'no prompt' provided SAM2 with a single bounding box covering the entire image - with the effective instruction of 'Find the most relevant object in this frame'.

### Test 2 - 3 Click Prompting
3 Click prompting provided 3 positive clicks to each 'object' in an image - each positive click was provided on the relevant object associated. An example is shown using the prompter - Where 3 positive points under 'Object 1' are added to the frame:
<img width="2457" height="1535" alt="image" src="https://github.com/user-attachments/assets/2f87149f-959e-41e3-8b94-e46b42078027" />

This methodology yielded the following results - For all cases in this file, split-15 is used to show the results:

![C_1_split_15_bbox_and_clicks_20251218_213109 (1)](https://github.com/user-attachments/assets/392e1423-b8c1-4ec3-a1d7-6d855c211bdd)


### Test 3 - Bounding Box  Prompting
Bounding Box prompting provided a single bounding box (With no positive or negative clicks) surrounding each object. With the effective instruction of 'identify the most prominant object in this frame'

### Test 4 - Combined Prompting
Combined prompting utilised both Test 2 and Test 3

## Discussion



