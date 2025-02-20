import pandas
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
import bokeh
import scipy
import numpy
from bokeh.plotting import output_notebook
#Creates path and filters player data so that at least 2000 minutes was played in that season
path = "AllStatsfrom2000.csv"
df = pandas.read_csv(path)
df["Total_Minutes"] = df["GP"] * df["MIN"]
df = df[df["Total_Minutes"]>=2000]
df["MIN"]

#Gets list of every column name in dataset
columns = df.columns.to_list()


#Extra column to remove
drop_columns = ["PLAYER_ID"]

#Gets rid of every column that ends with RANK or FANTASY
#Dropped because they are unnecessary columns
df = df[[c for c in df.columns if not (c.endswith('RANK') or "FANTASY" in c)]]
df = df.drop(drop_columns,axis=1)

#Gets columns for data on person. Not required for model
index_cols = ["PLAYER_NAME","SEASON", "W_PCT","NICKNAME","TEAM_ID","TEAM_ABBREVIATION","W","L","Total_Minutes"]


#Creates the ID as the index, and creates the personal data in its own dataset
player_columns = df[index_cols]
player_columns["ID"] = df.index
player_columns

#Drops the index Columns from original dataset
df = df.drop(index_cols,axis=1)

#Normalizes data to 0 and 1
df_non_normalized = df[:]
for column in df.columns.to_list():
    df[column] = (df[column] - df[column].min())/(df[column].max()-df[column].min())#Normalization

#Converts data and answers to numpy array (Like int arrays in Java and C++)
answers = player_columns["W_PCT"].to_numpy()
data = df.to_numpy()

#Splits Test and Train Data
total_train_percentage = 70
train_data, test_data, train_answers, test_answers = train_test_split(
      data, answers, train_size=total_train_percentage/100, random_state=69
  )
train_answers = train_answers.reshape(-1, 1)
test_answers = test_answers.reshape(-1, 1)


#CREATE AND COMPILE MODEL HERE. Please name Model "model"



#Plots Data
winning_percentages = list(model.predict(data))
winning_percentages = [list(i)[0] for i in winning_percentages]
winning_percentages = numpy.array(winning_percentages)


plots = []
for column in df.columns.tolist():
  column_vals = df_non_normalized[column].to_numpy()
  name = player_columns["PLAYER_NAME"].to_list()
  season = player_columns["SEASON"].to_list()
  nameAndSeason = [f"{name[i]}\n{season[i]}" for i in range(len(name))]
  slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(column_vals,winning_percentages)
  print(f"""
  {column} stats:
  Winning Percentage Change Per One: {slope*100}%
  r: {r_value}
  r^2: {r_value*r_value}
  standard deviation: {std_err}""")
  columns_vals = list(column_vals)
  winnings_percentages = list(winning_percentages)
  data = {'x': column_vals, 'y': winnings_percentages, 'labels': nameAndSeason}
  source = ColumnDataSource(data)

  # Create the plot
  p = figure(title="Interactive Plot", tools="pan,wheel_zoom,box_zoom,reset", tooltips=[("Label", "@labels"), ("(x, y)", "(@x, @y)")],x_axis_label=column,y_axis_label="Estimated Winning Percentage")
  p.scatter(x='x', y='y', size=10, source=source)
  plots.append(p)
grid = bokeh.layouts.column(plots)  # Arranged in two rows and two columns
show(grid)

