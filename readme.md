# Programming with Python Project - Weatherstation Data

This project was done for a python course. The python program can read data from a mysql database, a website supplying csv formatted data or a csv file and generate summaries and plots.

## Example usage

### Use as package

The program implements a new class "SensorData" which can fetch the data and generate summaries and plots as needed.

### CLI usage

A csv file with example data is provided in the repository. To generate output, simply clone the repository and run the following command:

```bash
python3.9 SensorData.py -m csv -f csv_example.csv -o ./html/weather -p ./html/plots.svg -t ./weather_template.html -c example.conf
```

Note: This will create a warning in the output file as the data from the csv file is outdated, and the displayed last n days are incorrect as it simply takes the whole csv file.

To generate output from a web or mysql source, simply edit the example.conf and run the following command:

```bash
python3.9 SensorData.py -m csv -d 14 -o ./html/weather -p ./html/plots.svg -t ./weather_template.html -c example.conf
```

Note that you need at least the columns "Time", "Temperature", "Humidity", "Pressure", "Air Quality", "Battery", "Illuminance" and "Sky Temperature" to properly run the program directly from CLI.
