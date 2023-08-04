# RecSys Challenge 2023

<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="header" />
</p>
<p align="center">
    <img src="https://i.imgur.com/MlLA8Hq.png" width="180" alt="Politecnico di Milano"/>
</p>

## Setup
To run the code you will need to have **Python 3.10** installed on your machine.

We recommend using a virtual environment to avoid conflicts with other projects.

To create a virtual environment, run the following command:
```bash
python3 -m venv .venv
```

To activate the virtual environment, run the following command:

- If you are using Linux: `source .venv/Scripts/activate`
- If you are using Windows, in the PowerShell: `.\.venv\Scripts\Activate.ps1`

To install the required dependencies, run the following command inside the virtual environment:
```bash
pip install -r requirements.txt
```
To run Catboost are required CUDA drivers, otherwise set GPU parameter to False.
## Running the code
Before running the code, import challenge's data in data folder.  
To run the code, run the following command inside the virtual environment:
```bash
python3 generate_predictions.py
```
Results are in predictions/final_predictions

## Team - "Gabibboost"
* [Alessandro Maranelli](https://github.com/alessandromaranelli)
* [Alessandro Verosimile](https://github.com/alessandroverosimile)
* [Andrea Riboni](https://github.com/andreariboni)
* [Arturo Benedetti](https://github.com/Benedart)
* [Davide Zanutto](https://github.com/davidezanutto)
* [Nicola Cecere](https://github.com/nicola-cecere)
* [Paolo Basso](https://github.com/paolobasso99)
* [Salvatore Marragony](https://github.com/salvatoremarragony)
* [Samuele Peri](https://github.com/john-galt-10)  

We worked under the supervision of:
* [Maurizio Ferrari Dacrema](https://github.com/maurizioFD)