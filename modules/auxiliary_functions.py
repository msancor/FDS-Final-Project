from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a dataframe and returns a dataframe with the new features that we will use in our prediction models

    Args:
        data (pd.DataFrame): The dataframe that we will use to create the new features

    Returns:
        pd.DataFrame: A dataframe with the new features
    """

    #First we create a copy of the dataframe in order to not modify the original one
    data = data.copy()

    #In order to define the attack zones we need to categorize the pitches according to their location in the plate
    #We define the attack zones as follows:
    #The heart zone is contained within the rule-book strike zone, its coordinates are between -0.558 and 0.558 in the x axis and between 1.833 and 3.166 in the z axis
    #The shadow zone is an area that covers the edge of the rule-book strike zone and several inches beyond the rule-book strike zone, its coordinates are between -1.108 and 1.108 in the x axis and between 1.166 and 3.833 in the z axis
    #The chase zone covers a zone where pitches that are definitively outside of the strike zone, its coordinates are between -1.666 and 1.666 in the x axis and between 0.5 and 4.5 in the z axis
    #Finally, the waste zone is the area that is outside of the chase zone, its coordinates are outside of the coordinates of the chase zone

    #Here we define the conditions for the attack zones and the values that will be assigned to each pitch according to the conditions
    conditions = [
        (data['plate_x'].between(-0.558, 0.558)) & (data['plate_z'].between(1.833, 3.166)),
        (data['plate_x'].between(-1.108, 1.108)) & (data['plate_z'].between(1.166, 3.833)),
        (data['plate_x'].between(-1.666, 1.666)) & (data['plate_z'].between(0.5, 4.5))
    ]
    #Here we define the values that will be assigned to each pitch according to the conditions
    values = ['heart', 'shadow', 'chase']
    #Here we create a new column in the dataframe that will contain the attack zone of each pitch
    data['attack_zone'] = np.select(conditions, values, default='waste')

    #Here we define a new feature that will contain the distance of each pitch from the center of the plate
    data['distance_from_center'] =  distance(data['plate_x'], 0, data['plate_z'], 2.5)

    #Here we define a new feature that will the total length of the strike zone on the vertical axis
    data['strike_zone_length'] = data['sz_top'] - data['sz_bot']

    #Here we define a new feature that will contain the total pitch movement from the center of the plate
    data['pitch_movement'] =  distance(data['pfx_x'], 0, data['pfx_z'], 0)

    #Here we define a new feature that will contain the total movement over average of each pitch
    data['movement_over_avg'] = data['pitch_movement'] - data.groupby('pitch_type')['pitch_movement'].transform('mean')

    #Here we define a new feature that will contain alignment of the pitcher and the hitter
    data['pitcher_hitter_alignment'] = data['p_throws'].astype(str) + data['stand'].astype(str)

    return data

def convert_to_numeric(data: pd.DataFrame, standardize: bool=True) -> pd.DataFrame:
  """
  This function takes a dataframe and returns a dataframe with the categorical features converted to numerical features

  Args:
      data (pd.DataFrame): The dataframe that we will use to create the new features
      standardize (bool): If True, the numerical features will be standardized

  Returns:
      pd.DataFrame: A dataframe with the categorical features converted to numerical features
  """

  #First we create a copy of the dataframe in order to not modify the original one
  data = data.copy()

  #We convert boolean columns to int to obtain numerical features 0 and 1
  data['on_3b'] = data[['on_3b']].astype('int32')
  data['on_2b'] = data[['on_2b']].astype('int32')
  data['on_1b'] = data[['on_1b']].astype('int32')

  #Here we convert the stand feature to a binary feature
  data['stand'] = data['stand'].apply(lambda x: 1 if x == 'R' else 0)
  #Here we rename the stand feature to right_stand
  data.rename(columns={'stand': 'right_stand'}, inplace=True)

  #We do the same with the p_throws feature
  data['p_throws'] = data['p_throws'].apply(lambda x: 1 if x == 'R' else 0)
  data.rename(columns={'p_throws': 'right_p_throws'}, inplace=True)

  #We do the same with the inning_topbot feature
  data['inning_topbot'] = data['inning_topbot'].apply(lambda x: 1 if x == 'Top' else 0)
  data.rename(columns={'inning_topbot': 'top_inning'}, inplace=True)

  #Here we performe one hot encoding for the remaining categorical features
  data = pd.get_dummies(data, dtype='int32')

  #Here we standardize the numerical features if the standardize parameter is set to True
  if standardize:
    #Here we define the columns that we want to standardize
    continous_columns = ['sz_top', 'sz_bot', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'inning',
                         'outs_when_up', 'balls', 'strikes', 'release_speed', 'spin_axis', 'release_spin_rate',
                         'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'distance_from_center', 'strike_zone_length',
                         'pitch_movement', 'movement_over_avg'] 
    
    #Here we create a StandardScaler object
    scaler = StandardScaler()
    #Here we scale only the columns in the continous_columns list
    data[continous_columns] = scaler.fit_transform(data[continous_columns])

  #Finally we reorder the columns so that the target variable is the last one
  data = data[[c for c in data if c not in ['is_strike']] + ['is_strike']]

  return data
     

def distance(x1: float, x2: float, z1: float, z2: float) -> float:
    """
    This function takes the Euclidean distance between two points in a 2D space

    Args:
        x1 (float): The x coordinate of the first point
        x2 (float): The x coordinate of the second point
        z1 (float): The z coordinate of the first point
        z2 (float): The z coordinate of the second point

    Returns:
        float: The Euclidean distance between the two points
    """
    #Here we calculate the Euclidean distance between the two points
    return np.sqrt(((x2 - x1)**2) + ((z2 - z1)**2))

def draw_sz(sz_top: float=3.5, sz_bot: float=1.5, ls: str='k-') -> None:
  """
  This function draws the average strike-zone for batters in our dataset.
  This function was taken from the Kaggle notebook of user @nickwan in order to visualize the strike-zone in our plots.
  We don't take credit for this function and we don't claim it as our own.

  Args:
    sz_top (float): the top of the strike-zone
    sz_bot (float): the bottom of the strike-zone
    ls (str): the linestyle (use `plt.plot()` linestyle conventions)
  """
  plt.plot([-0.708, 0.708], [sz_bot,sz_bot], ls)
  plt.plot([-0.708, -0.708], [sz_bot,sz_top], ls)
  plt.plot([0.708, 0.708], [sz_bot,sz_top], ls)
  plt.plot([-0.708, 0.708], [sz_top,sz_top], ls) 

def draw_home_plate(catcher_perspective: bool=True, ls: str='k-'):
  """
  This function draws the home plate from either the catcher perspective or pitcher perspective.
  This function was taken from the Kaggle notebook of user @nickwan in order to visualize the strike-zone in our plots.
  We don't take credit for this function and we don't claim it as our own. 

  Args:
    catcher_perspective (bool): if True, draws the home plate from the catcher perspective. If False, draws the home plate from the pitcher perspective.
    ls (str): the linestyle (use `plt.plot()` linestyle conventions)
  """
  if catcher_perspective:
    plt.plot([-0.708, 0.708], [0,0], ls)
    plt.plot([-0.708, -0.708], [0,-0.3], ls)
    plt.plot([0.708, 0.708], [0,-0.3], ls)
    plt.plot([-0.708, 0], [-0.3, -0.6], ls)
    plt.plot([0.708, 0], [-0.3, -0.6], ls)
  else: 
    plt.plot([-0.708, 0.708], [0,0], ls)
    plt.plot([-0.708, -0.708], [0,0.1], ls)
    plt.plot([0.708, 0.708], [0,0.1], ls)
    plt.plot([-0.708, 0], [0.1, 0.3], ls)
    plt.plot([0.708, 0], [0.1, 0.3], ls)

def draw_attack_zones():
  """
  This function draws the Statcast attack zones on a plot.
  This function was taken from the Kaggle notebook of user @nickwan in order to visualize the strike-zone in our plots.
  We don't take credit for this function and we don't claim it as our own.
  """

  # outer heart / inner shadow
  plt.plot([-0.558, 0.558], [1.833,1.833], color=(227/255, 150/255, 255/255), ls='-', lw=3)
  plt.plot([-0.558, -0.558], [1.833,3.166], color=(227/255, 150/255, 255/255), ls='-', lw=3)
  plt.plot([0.558, 0.558], [1.833,3.166], color=(227/255, 150/255, 255/255), ls='-', lw=3)
  plt.plot([-0.558, 0.558], [3.166,3.166], color=(227/255, 150/255, 255/255), ls='-', lw=3) 

  # outer shadow /  inner chase 
  plt.plot([-1.108, 1.108], [1.166,1.166], color=(255/255, 197/255, 150/255), ls='-', lw=3)
  plt.plot([-1.108, -1.108], [1.166,3.833], color=(255/255, 197/255, 150/255), ls='-', lw=3)
  plt.plot([1.108, 1.108], [1.166,3.833], color=(255/255, 197/255, 150/255), ls='-', lw=3)
  plt.plot([-1.108, 1.108], [3.833,3.833], color=(255/255, 197/255, 150/255), ls='-', lw=3) 

  # outer chase 
  plt.plot([-1.666, 1.666], [0.5,0.5], color=(209/255, 209/255, 209/255), ls='-', lw=3)
  plt.plot([-1.666, -1.666], [0.5,4.5], color=(209/255, 209/255, 209/255), ls='-', lw=3)
  plt.plot([1.666, 1.666], [0.5,4.5], color=(209/255, 209/255, 209/255), ls='-', lw=3)
  plt.plot([-1.666, 1.666], [4.5,4.5], color=(209/255, 209/255, 209/255), ls='-', lw=3) 
