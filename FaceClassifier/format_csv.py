import pandas as pd
import numpy as np
import os
import argparse

def format_save_csv(label_dir,save_dir):
  """
  Formats and parses the csv files to get the files that have AU data in them, otherwise discards them. Discarded filenames are
  saved in a file called discard.txt. Also saves the formatted csv files with the same names in the save_dir directory.

  Arguments:
  - label_dir: str, directory that contains the original csv files/
  - save_dir: str, directory to which the formatted csv files will be written to. If it doesn't exist, one will be created.

  Returns
  Nothing, zilch, nada.
  """
  if not os.path.isdir(save_dir):
    print('Destination directory {} does not exist, creating one now...'.format(save_dir))
    os.makedirs(save_dir)
  discardCount = 0
  discardFile = 'discard.txt'
  labels = [file for file in os.listdir(label_dir) if file.endswith('.csv')]
  print('Found {} csv files in {}'.format(len(labels),label_dir))
  file = open(discardFile,'w')
  for index in range(len(labels)):
    # print('Reading in {}'.format(labels[i]))
    # read csv
    df = pd.read_csv(os.path.join(label_dir,labels[index]))
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
      # print('Key not found, discarding {}'.format(labels[index]))
      file.write('{}\n'.format(labels[index]))
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
        #master.to_csv('{}'.format(os.path.join(save_dir,labels[index])),index=False)
        pd.DataFrame(list(zip(list(finaldict.keys()),list(finaldict.values()))),columns=['Time','Labels']).to_csv('{}'.format(os.path.join(save_dir,labels[index])),index=False)
    else:
      file.write('{}\n'.format(labels[index]))
      discardCount+=1
  file.close()
  print('Discarded a total of {} files, filenames are available in {}'.format(discardCount,discardFile))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv_dir', default='labels/', type=str, 
                      help="Path to the original csv files directory")
  parser.add_argument('--save_dir', default='save/', type=str, 
                      help="Directory where formatted csv files will be saved.\n" 
                      "If directory does not exist, one will be created")
  args = parser.parse_args()
  format_save_csv(args.csv_dir,args.save_dir)
