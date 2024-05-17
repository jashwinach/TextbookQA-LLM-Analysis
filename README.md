BLIP-2 can be setup through this repository: [LAVIS](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).

LLaVA can be setup through this repository: [LLaVA](https://github.com/haotian-liu/LLaVA).

If you get any package issues with BLIP-2 or LLaVA, the best option is to clone both repos above and place the blip2.py file in the ./blip2 folder under the root of the BLIP-2 git directory and place the llava.py file in the ./llava folder under the root of the LLaVA git directory.

For GPT-4, you have to create an account on the GPT-4 API website and create an API key. Helpful link: [GPT-4](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo).

Folder Structure:

<ul>
    <li>
    ./Dataset -> This folder contains our train, test and validation folders and Dataset. For the purpose of our work we only focus on the test set for zero-shot evaluation.
    </li>
    <li>
    ./Dataset/test/tqa_v2_test.json -> This contains the test data which is parsed using the `prepare_test_data.ipynb` file to create three CSV files: 
        <ul>
            <li>DiagramQuestionsData.csv: This contains images and associated MCQs.</li>
            <li>NonDiagram_MCQ_QuestionsData.csv: This contains regular MCQs with no associated images.</li>
            <li>NonDiagram_True_False_QuestionsData.csv: This contains regular True/False questions with no associated images.</li>
        </ul>
    </li>
    <li>BLIP-2 files:
        <ul>
            <li>./blip2/blip2.py: This file can be run using the command `python blip2.py` and generates predictions using BLIP-2 on images and questions defined in the ./Dataset/test/DiagramQuestionsData.csv file.</li>
            <li>./blip2/blip2_answers.csv: This file contains all the BLIP-2 generated answers for Diagram MCQs</li>
            <li>./blip2/common_blip2_llava_gpt4_preds.csv: This file contains the small subset of images and questions along with image descriptions.</li>
        </ul>
    </li>
    <li>LLaVA files:
        <ul>
            <li>./llava/llava.py: This file can be run using the command `python llava.py` and generates predictions using LLaVA on images and questions defined in the ./Dataset/test/DiagramQuestionsData.csv file.</li>
            <li>./llava/llava_7B_answers.csv: This file contains all the LLaVA generated answers for Diagram MCQs</li>
            <li>./llava/common_blip2_llava_gpt4_preds.csv: This file contains the small subset of images and questions along with image descriptions.</li>
        </ul>
    </li>
    <li>GPT-4 files:
        <ul>
            <li>./GPT4/GPT4.ipynb: This jupyter notebook contains code for generating predictions using GPT-4 for Diagram MCQs, Non-diagram MCQs and True/False questions.</li>
            <li>./GPT4/GPT_4_preds.csv: This file contains GPT-4 generated answers for Diagram MCQs. The GPT-4 predictions for Non-diagram MCQs and True/False questions are stored in ./Dataset/test/NonDiagram_MCQ_QuestionsData.csv and ./Dataset/test/NonDiagram_True_False_QuestionsData.csv files.</li>
        </ul>
    </li>
    <li>Analysis files: All the CSV files in this folder contain correct and wrong predictions for our VLMs for manual inspection of images and questions they struggled with.
        <ul>
            <li>Analysis.ipynb: This jupyter notebook contains code for manually judging image and question complexity for Diagram MCQs as well as Non-diagram MCQs and True/False questions.</li>
            <li>PlotGraphs.ipynb: This jupyter notebook contains code for plotting our accuracy graphs for Diagram MCQs, Non-diagram MCQs and True/False questions. <li>
        </ul>
    </li>
</ul>

Our report contains details about our analysis and results.