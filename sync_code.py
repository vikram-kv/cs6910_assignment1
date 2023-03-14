import os, sys, time

while True:
    try:
        time.sleep(150)
        os.system('wandb sync --sync-all')
    except KeyboardInterrupt:
        print('Quitting')
        sys.exit(0)