import os
from pathlib import Path

def clean_logs(max_lines=1000):

  this_files_Path = Path(os.path.realpath(__file__))
  log_files = [this_files_Path.parent / f 
              for f in os.listdir(this_files_Path.parent)
              if '.log' in f]

  for log_file in log_files:
      lines = [line for line in open(log_file)]
      with open(log_file, 'w') as f:
          f.writelines(lines[-max_lines:])
      
if __name__=='__main__':
  clean_logs()