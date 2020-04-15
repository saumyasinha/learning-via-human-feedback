import pandas as pd
import numpy as np
import os
import argparse

def get_unique_labels(csv_dir):
  """
  Returns a list of unique labels in a directory of csv files since each csv file has a different list of labels
  """
  csv_files = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]
  all_cols = []
  for csv_file in csv_files:
    df = pd.read_csv(os.path.join(csv_dir,csv_file))
    au_cols = ['AU' in col_name for col_name in df.columns]
    audf = df[list(np.array(df.columns)[au_cols])]
    all_cols.append(audf.columns.to_list())
  unique_labels = list(set([item for sublist in all_cols for item in sublist]))
  return unique_labels


def add_AU_columns(df, map_dict):
  """
  Adds AU columns back to the dataframe but this time, each df will be uniform as they will all have the same columns.
  Presence of an AU is indicated by 1, absence is indicated by 0
  """
  # add separate columns for each AU and initialize them to 0
  df = df.assign(**map_dict)
  df.iloc[:,2:] = 0

  # assign 1 to AU column if that AU exists in the "Labels" column
  for row_index in df.index:
    for column_name in df.iloc[row_index]['Labels']:
      df.at[row_index,column_name] = 1
  return df


def format_csv_files(original_csv_dir, save_dir):
  """
  Formats and parses the csv files to get the files that have AU data in them, otherwise discards them. Discarded filenames are
  saved in a file called discard.txt. Also saves the formatted csv files with the same names in the save_dir directory.

  Args:
  - original_csv_dir: str, directory that contains the original csv files/
  - save_dir: str, directory to which the formatted csv files will be written to. If it doesn't exist, one will be created.

  Returns:
    Doesn't return anything but saves csv files with columns Time, Labels, AU00, AU01, ... AU50
  """
  if not os.path.isdir(save_dir):
    print('Destination directory "{}" does not exist, creating one now...'.format(save_dir))
    os.makedirs(save_dir)
  discardCount = 0
  discardFile = 'discard.txt'
  unique_labels = get_unique_labels(original_csv_dir)
  map_dict = {}
  for count,i in enumerate(unique_labels):
    map_dict[i] = count

  csv_filenames = [file for file in os.listdir(original_csv_dir) if file.endswith('.csv')]
  print('Found {} csv files in {}'.format(len(csv_filenames),original_csv_dir))
  file = open(discardFile,'w')
  for csv_name in csv_filenames:
    # print('Reading in {}'.format(csv_name))
    # read csv
    df = pd.read_csv(os.path.join(original_csv_dir,csv_name))
    # get the columns that have "AU" in them
    au_cols = ['AU' in col_name for col_name in df.columns]

    # new dataframe that only has time and the AU columns
    audf = df[['Time']+list(np.array(df.columns)[au_cols])]

    # Threshold to get columns which have at least 1 value >=thresh
    thresh = 0.01
    audf = audf.loc[:, audf.ge(thresh).any()]
    try:
      # Get seconds as integers
      audf['Seconds']=audf['Time'].astype(int)
    except KeyError:
      # print('Key not found, discarding {}'.format(csv_name))
      file.write('{}\n'.format(csv_name))
      discardCount+=1
      continue

    # master dataframe to finally save
    master = pd.DataFrame([])

    # group the data by the time and take only the mean of the data for each second
    for timecode in np.unique(audf['Seconds'].to_numpy()):
      temp = np.mean(audf[audf['Seconds']==timecode],axis=0)
      temp = pd.DataFrame(temp).transpose()
      master = master.append(temp)

    master = master.reset_index(drop=True)
    cols = list(master.columns)
    # change order of columns to have time, seconds, au01, au02,...
    cols.insert(1, cols.pop(cols.index('Seconds')))

    # Don't save dataframes that don't have more than 2 columns (time and seconds columns)
    # I'm sure there's a better way to avoid this earlier in the code but I'm tired of looking at these csv files
    if len(cols)>2:
      master = master[cols] 
      # drop any zero rows
      master = master[(master.iloc[:,2:].T != 0).any()]
      aus = master.iloc[:,2:]
      finaldict = {}
      for idx,rows in aus.iterrows():
        finaldict[master['Seconds'][idx]] = pd.DataFrame(rows[rows != 0]).transpose().columns.to_list()
        
      saving_df = pd.DataFrame(list(zip(list(finaldict.keys()),list(finaldict.values()))),columns=['Time','Labels'])
      saving_df = add_AU_columns(saving_df,map_dict)
      # drop frame at time=0 if it exists because there are multiple files having empty images at t=0
      if saving_df['Time'][0]==0:
        saving_df = saving_df.drop([0])
      saving_df.to_csv('{}'.format(os.path.join(save_dir,csv_name)),index=False)
    else:
      file.write('{}\n'.format(csv_name))
      discardCount+=1
  file.close()
  print('Discarded a total of {} files, discarded filenames are available in {}'.format(discardCount,discardFile))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv_dir', default='labels/', type=str, 
                      help="Path to the original csv files directory")
  parser.add_argument('--save_dir', default='save/', type=str, 
                      help="Directory where formatted csv files will be saved.\n" 
                      "If directory does not exist, one will be created")
  args = parser.parse_args()
  format_csv_files(args.csv_dir, args.save_dir)
  print('Completed CSV formatting, they are available in {}'.format(args.save_dir))