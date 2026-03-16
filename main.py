from pandas._libs.hashtable import mode
import tracemalloc
from dotenv import load_dotenv
from src.util import setup_logging
from src.graph import run_debate
logger = setup_logging('main')

def main(): 
    load_dotenv()
    tracemalloc.start()
  
    run_debate('AAPL', mode='debate')
        

if __name__ == '__main__': 
    main()