import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import argparse
import configparser

class SensorData:
    """
    This class will fetch data from an URL, csv, or directly connect to a mysql database.
    
    It can construct some analysis of the fetched data and plot it.
    Output is html.
    
    The first column of the data needs to be the timestamp.
    
    Usage:
    create new SensorData object
    fetch data with fetch function
    create summaries and plots with the other functions
    """
    
    def __init__(self, connection_type, url="", apikey="", file="", sql_credentials = []):
        """
        SensorData class constructor
        
        connection_type: "web", "sql" or "csv"
        url: URL to the site to fetch data from (only for web)
        apikey: will be sent as GET request as "apikey" (only for web)
        file: csv file to use (only for csv)
        sql_credentials: array of host, database, user, password, table and for the mysql database (only for sql)
        """

        if(connection_type not in ["web", "sql", "csv"]):
            raise ValueError("connection_type must be one of 'web', 'csv' or 'sql'")
        
        self.connection_type=connection_type
        self.url = url
        self.apikey = apikey
        self.file=file
        self.sql_credentials = sql_credentials
        
    def fetch(self, last_n_days, time_column="time", col_names=[], col_units=[]):
        """
        Fetches the data from the specified connection
        
        last_n_days: number of last days to fetch
        time_column: name of the time column
        col_names: array containing new column names. If left empty, default column names will be used
        col_units: array containing units for the columns. If left empty, units will be blank
        """
        
        self.fetch_timestamp = datetime.datetime.now()
        self.last_n_days = last_n_days
        
        if(self.connection_type == "web"):
            #Fetch data from web
            request_url = self.url + "?apikey=" + self.apikey + "&days=" + str(last_n_days)
            self.df = pd.read_csv(request_url, skipinitialspace=1)
            self.df[time_column] = pd.to_datetime(self.df[time_column])
        
        elif(self.connection_type == "csv"):
            self.df = pd.read_csv(self.file, skipinitialspace=1)
            self.df[time_column] = pd.to_datetime(self.df[time_column])
            
        elif(self.connection_type == "sql"):
            import mysql.connector as connector
            
            db = connector.connect(
                host=self.sql_credentials[0],
                database=self.sql_credentials[1],
                user=self.sql_credentials[2],
                passwd=self.sql_credentials[3]
            )
            
            query = "SELECT * FROM {table} WHERE {time_field} > DATE_SUB(now(), INTERVAL {days} DAY);".format(table=self.sql_credentials[4], time_field=self.sql_credentials[5], days=self.last_n_days)
            
            try:
                self.df = pd.read_sql(query, db)
                db.close()
            except Exception as e:
                db.close()
                return "Error fetching from database:{}".format(e)

        
        #Input verification
        if(col_names != [] and len(col_names) != len(self.df.columns)):
            raise ValueError("colnames must be empty or the same length as columns.")

        if(col_units != [] and len(col_units) != len(self.df.columns)):
            raise ValueError("colnames must be empty or the same length as columns.")
        
        #Rename columns, save units
        if(col_names != []):
            self.df.columns = col_names
        
        if(col_units != []):
            self.units = dict(zip(self.df.columns, col_units))
        else:
            self.units = dict(zip(self.df.columns, np.full([len(self.df.columns)], "")))
            
    def mutate(self, column, mutation):
        """
        Some columns need additional work to display the correct unit.
        
        column: string of column to mutate
        mutation: string, function of x apply to the column e.g. "x / 500 + 3"
        """
        x = self.df[column]
        self.df[column] = eval(mutation)
        
    def summary(self, columns=[]):
        """
        Produces a short summary in the form of:
        
        The current *colname* is *current_value* *col_unit*. Over the last *last_n_days*, the minimum *colname* was ... measured at *time* and the maximum was ... measured at *time*
        
        columns: string or array of column names to produce a summary of
        """
        
        summaries = []
            
        for col in columns:
            summary = "The current {colname} is {current_value:.2f}{unit} measured at {current_time}.<br>Over the last {n_days} days, the minimum {colname} was {min_value:.2f}{unit} measured on {min_time} and the maximum was {max_value:.2f}{unit} on {max_time}.<br><br>"
            
            min_index = self.df[col].idxmin()
            max_index = self.df[col].idxmax()
            
            summary = summary.format(colname = col.lower(), current_value = self.df.iloc[-1][col], current_time=self.df.iloc[-1][self.df.columns[0]], unit = self.units[col], n_days = self.last_n_days, min_value = self.df.at[min_index, col] , min_time = self.df.at[min_index, self.df.columns[0]], max_value = self.df.at[max_index, col], max_time = self.df.at[max_index, self.df.columns[0]])
            
            summaries.append(summary)
        
        summaries = "".join(summaries)
        summaries = "<div class='summary'>" + summaries + "</div>"
        
        return summaries
    
    def status(self, update_interval_s = 300, battery_column="Battery", battery_low_level=3.2, update_interval_warning_factor = 2):
        """
        Returns the current status of the weatherstation. This includes checking the battery level, and when the last update was

        update_interval_s: time interval in seconds the weatherstation should normally update
        battery_column: name of the battery column
        battery_low_level: battery level below which a low battery warning is displayed
        update_interval_warning_factor: factor by which the update interval will be multiplied to set as threshold for a warning message
        """
        
        statusmsg = ""
        warning = False
        
        time_diff = self.fetch_timestamp - self.df.iloc[-1]["Time"]
        
        if(time_diff.seconds > update_interval_s * update_interval_warning_factor):
            statusmsg += "<em style='color:green;'>WARNING:</em> I have not recieved data since {last_update}, which was {min_ago:.1f} minutes before this report was generated!<br>".format(last_update=self.df.iloc[-1]["Time"], min_ago = time_diff.seconds/60)
            warning = True
            
        if(self.df[battery_column].iloc[-1] < battery_low_level):
            statusmsg += "<em style='color:red;'>WARNING:</em> Battery level is low!<br>"
            warning = True
            
        if(not warning):
            statusmsg = "<em style='color:green;'>Everything seems to be OK</em><br>" + statusmsg
            
        statusmsg = "Data fetched on {fetch_timestamp}<br>Last update revieved on {last_update} ({min_ago:.1f} minutes before this report)<br>".format(fetch_timestamp = self.fetch_timestamp.strftime("%Y-%m-%d %H:%M:%S"), last_update=self.df.iloc[-1]["Time"], min_ago = time_diff.seconds/60) + statusmsg + "<br><br>"
        
        statusmsg = "<div class='status'>" + statusmsg + "</div>"
        
        return statusmsg
    
    def cloudPrediction(self, ambient_temp_col="Temperature", sky_temp_col="Sky Temperature", threshold=(-8, 3)):
        """
        Tries to inherit wether it is cloudy or not by comparing the temperature of the sky (measured eg. with an infrared thermometer) to the ambient temperature.
        
        ambient_temp_col: column in the dataframe where the ambient temperature is stored
        sky_temp_col: column in the dataframe where the sky temperature is stored
        threshold: touple of thresholds below and above the sky is supposedly clear. Delta is calculated as ambient-sky temperature. Negative delta is observed at clear day, positive delta at clear night.
        """
        delta = self.df[ambient_temp_col].iloc[-1] - self.df[sky_temp_col].iloc[-1]
        
        if(delta <= threshold[0] or delta >= threshold[1]):
            cloudmsg = "It is probably not cloudy (delta is {:.1f}K).<br>".format(delta)
        else:
            cloudmsg = "It is probably cloudy (delta is {:.1f}K)<br>".format(delta)
        
        cloudmsg = "<div class='cloudmsg'>" + cloudmsg + "</div>"
        return cloudmsg
        
    def simplePlot(self, column, ax="", title="", color="black"):
        """
        Returns a simple time plot for the given columns as seaborn plot
        
        column: column to plot
        ax: subplots axis object
        title: title of the plot. Requires defining an ax object
        color: color of the line
        """
        
        if(ax != ""):
            ax.set(title=title, ylabel=self.units[column])
            sns.lineplot(ax=ax, data=self.df, x=self.df.columns[0], y=column, color=color)
            ax.grid(True, linewidth=0.5, color="gray")
            return True
        else:
            sns.lineplot(data=self.df, x=self.df.columns[0], y=column, color=color)
            return True
    
    def rollingAverage(self, column, new_column, n_rolling=5):
        """
        Creates a new column containing the rolling average across the last n_rolling entries

        column: column name to calculte the rolling average of
        new_column: column name of the new column containing the rolling average
        n_rolling: last n measurements to base the rolling average on 
        """

        self.df[new_column] = self.df[column].rolling(n_rolling).mean()
        self.units[new_column] = self.units[column]

        return True
        
    def dailyMinMaxPlot(self, column, ax="", title="", colors=["blue", "red"]):
        """
        Returns a plot with the daily min and max values for the given column
        
        column: column to plot
        ax: subplots axis object
        title: title of the plot. Requires defining an ax object
        color: array of colors for min[0] and max[1] 
        """
        min_values = self.df.groupby(self.df[self.df.columns[0]].dt.date).min()[column]
        max_values = self.df.groupby(self.df[self.df.columns[0]].dt.date).max()[column]
        
        if(ax != ""):
            ax.set(title=title, ylabel=self.units[column])
            sns.lineplot(ax=ax, x=max_values.index, y=max_values.values, color=colors[1], marker="x")
            sns.lineplot(ax=ax, x=min_values.index, y=min_values.values, color=colors[0], marker="x")
            ax.legend(["max", "min"])
            ax.grid(True, linewidth=0.5, color="gray")
        else:
            sns.lineplot(x=max_values.index, y=max_values.values, color=colors[1], marker="x")
            sns.lineplot(x=min_values.index, y=min_values.values, color=colors[0], marker="x")

    def heatmap(self, columns, ax="", title="Heatmap of Standard Scores", labels=[]):
        """
        Generates a heatmap for all columns in columns ordered by time of the day.
        Performs mean standard score normalization ( (x-mean(x))/std(x) ) to appropiately display values on different scales.

        columns: array of columns to include in the heatmap
        ax: subplots axis object (optional)
        title: title of the plot (optional)
        labels: array of labels for the columns (optional)
        """
        if (labels != [] and len(labels) != len(columns)):
            raise ValueError("Amount of labels need to math amount of given columns")
        elif(labels == []):
            labels = columns

        hourly_mean = self.df.groupby(self.df[self.df.columns[0]].dt.hour).mean()
        hourly_mean = hourly_mean[columns]
        hourly_mean_norm = (hourly_mean - hourly_mean.mean()) / hourly_mean.std()

        if(ax != ""):
            ax.set(title=title)
            sns.heatmap(hourly_mean_norm.transpose(), yticklabels=labels, ax=ax, cbar_kws={"label": "Standard Score"}, center=0, cmap="BuPu")
        else:
            sns.heatmap(hourly_mean_norm.transpose(), yticklabels=labels, cbar_kws={"label": "Standard Score"}, center=0, cmap="BuPu")

if(__name__ == "__main__"):

    parser = argparse.ArgumentParser(description="Program that generates nice html output from weather data.\nNote: csv method will always use the whole csv file and the displayed n last days are not correct!")
    parser.add_argument('-m', '--method', help="Set the connection type 'web', 'csv' or 'sql'. ", required=True, type=str)
    parser.add_argument('-t', '--template', help="Path to the html template", required=False, default="./weather_template.html", type=str)
    parser.add_argument('-o', '--outfile', help="Output file to write the generated html to.", required=False, type=str, default="./weather.html")
    parser.add_argument('-p', '--plotfile', help="File the plot should be written to.", required=False, type=str, default="./plots.svg")
    parser.add_argument('-d', '--days', help="Number of last n days to fetch data from. Default is 14", required=False, default=14, type=int)
    parser.add_argument('-c', '--credentials', help=".conf file containing credentials for web and sql method as well as column names and units. See example.conf for more information", required=True, type=str)
    parser.add_argument('-f', '--csvfile', help="Path to CSV file if using method 'csv'", required=False, type=str)

    args = parser.parse_args()

    if(args.method == "web"):
        config = configparser.RawConfigParser()
        config.read(args.credentials)
        col_names = config['web']['col_names'].split(",")
        col_units = config['web']['col_units'].split(",")

        con = SensorData("web", url=config['web']['url'], apikey=config['web']['apikey'])
        con.fetch(args.days, col_names=col_names, col_units=col_units)

    elif(args.method == "sql"):
        config = configparser.RawConfigParser()
        config.read(args.credentials)
        col_names = config['sql']['col_names'].split(",")
        col_units = config['sql']['col_units'].split(",")

        con = SensorData("sql", sql_credentials=[config['sql']['host'] , config['sql']['db'], config['sql']['user'], config['sql']['pw'], config['sql']['table'], config['sql']['time_column']])
        con.fetch(args.days, col_names=col_names, col_units=col_units )

    elif(args.method == "csv"):
        config = configparser.RawConfigParser()
        config.read(args.credentials)
        col_names = config['csv']['col_names'].split(",")
        col_units = config['csv']['col_units'].split(",")

        con = SensorData("csv", file=args.csvfile)
        con.fetch(args.days, col_names=col_names, col_units=col_units)

    # Generate output:

    #Derive columns
    con.mutate("Battery", "(x+2)/135")
    con.mutate("Air Quality", "x/1000")
    con.mutate("Illuminance", "np.log10(x)")

    #Plot data
    plt.style.use("dark_background")
    plt.rcParams["font.family"] = "monospace"
    fig, axes = plt.subplots(5,2, figsize=(20,20))

    con.simplePlot(column="Temperature", title="Temeprature", ax=axes[0,0], color="white")
    con.simplePlot(column="Humidity", title="Humidity", ax=axes[0,1], color="white")
    con.dailyMinMaxPlot(column="Temperature", title="Daily min/max Temperature", ax=axes[1,0], colors=["blue", "red"])
    con.dailyMinMaxPlot(column="Humidity", title="Daily min/max Humidity", ax=axes[1,1], colors=["blue", "red"])

    con.simplePlot(column="Pressure", title="Air Pressure", ax=axes[2,0], color="white")
    con.rollingAverage("Wind", "Wind rolling", 5)
    con.simplePlot(column="Wind rolling", title="Wind Speed (average over last 5 measurements)", ax=axes[2,1], color="white")
    con.simplePlot(column="Air Quality", title="Air Quality", ax=axes[3,1], color="white")
    con.simplePlot(column="Illuminance", title="Illuminance", ax=axes[3,0], color="white")
    con.dailyMinMaxPlot(column="Illuminance", title="Daily min/max Illuminance", ax=axes[4,0], colors=["darkblue", "white"])
    con.heatmap(columns=["Temperature", "Humidity", "Pressure","Wind","Air Quality", "Illuminance"], ax=axes[4,1])

    fig.tight_layout()
    fig.savefig(args.plotfile)

    #Write output to html
    f = open(args.outfile, "w")

    template = open(args.template, "r").read()

    status = con.status()
    summary = con.summary(["Temperature", "Humidity", "Air Quality"])
    cloud_pred = con.cloudPrediction()

    html_code = template.format(status=status, summary=summary, cloud_pred=cloud_pred)

    f.write(html_code)
    f.close()
