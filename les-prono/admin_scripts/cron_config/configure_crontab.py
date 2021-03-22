import logging

from crontab import CronTab

'''Probably is easier to hand configure a crontab file,
but this is made in order to help those that don't know
how to do that, to select the appropiate environment,
and to automatically initialize after a reboot'''

def configure_crontab(dcfg):
    verbose = dcfg['crontab']['verbose']

    # crontab configuration
    cronfile = CronTab()
    cronfile = CronTab(tabfile='cron_config/custom_crontab.tab')
    ccfg = dcfg['crontab']
    logging.debug('Load custom_crontab.tab')

    if verbose:
        print('crontab configuration:')
        print('<<<<<<<<<<')
        print('previous conf:')
        for job in cronfile:
            print(job)
            print("Is enabled:", job.is_enabled())

    # clean previous conf
    if ccfg['clear_all']:
        cronfile.remove_all()
        logging.debug('Remove past crontab entries')

    # crontab config: disable existing jobs
    for job in cronfile:
        job.enable(False)
        logging.debug('Disable past crontab entries')

    prepare = f"cd {dcfg['paths']['project']} && "\
            ". les-venv/bin/activate && "
    run_python_module = 'python -m '

    # crontab config: repeat this file you are reading
    if ccfg['automatic_restart']:
        job_restart = cronfile.new(
            command= prepare + run_python_module + "admin_scripts.start",
            comment='Run start script')
        job_restart.every_reboot()
        logging.debug('Set start.py to execute every reboot')

    # crontab config: first job
    pppm = ccfg['preprocessing_period_in_minutes']
    job_preprocessing = cronfile.new(
        command= prepare + run_python_module +"preprocessing.run_preprocessing",
        comment='Run preprocessing')
    job_preprocessing.setall(
        f'*/{pppm} * * * *'
        )  # useful: https://crontab.guru/
    logging.debug(f'Set run_preprocessing.py to execute every {pppm} min')

    # crontab config: clean_logs job
    job_clean_logs = cronfile.new(
        command= prepare +  run_python_module +"admin_scripts.logs.clean_logs",
        comment='Run clean_logs.py')
    job_clean_logs.setall(
        f'2 */2 * * *'
        )  # useful: https://crontab.guru/
    logging.debug(f'Set clean_logs.py to execute every two hours')

    # crontab config: algorithms' manager job
    apm = ccfg['algorithms_period_in_minutes']
    job_manager = cronfile.new(
        command= prepare +  run_python_module +"algorithms.manager" + " && rsync -rav /home/franchesoni/les-prono/predictions/ worker_goes@164.73.222.53:~/solar/sat/PRS/dev/PRS-sat/pronostico/estaciones/",
        comment='Run manager.py')
    job_manager.setall(
        f'*/{apm} * * * *'
        )  # useful: https://crontab.guru/
    logging.debug(f'Set manager.py to execute every {apm} min')

    # crontab config: predictions' sender job
    spm = ccfg['sender_period_in_minutes']
    job_sender = cronfile.new(
        command= prepare +  run_python_module +"algorithms.sender",
        comment='Run sender.py')
    job_sender.setall(
        f'*/{spm} * * * *'
        )  # useful: https://crontab.guru/
    logging.debug(f'Set sender.py to execute every {spm} min')

    # sync ssat02 onto ssat01
    job_sync_stack = cronfile.new(
      command="rsync --delete --exclude='CNT/' -au franchesoni@164.73.222.53:/solar/sat/stack-art/ART_G020x020RR_C020x020/ stack_mirror/")
    job_sync_stack.setall(
      '* * * * *'
    )


    # see crontab env:
    # job_env = cronfile.new(command="env > ~/cronenv")

    # crontab config: save changes
    cronfile.write()
    cronfile.write_to_user(user=dcfg['crontab']['user'])
    logging.debug('Write custom_crontab.tab')
    logging.debug("Set custom_crontab.tab as user's crontab")

    if verbose:
        print('>>>>>>>>>>')
        print('final conf:')
        for job in cronfile:
            print(job)
            print("Is enabled:", job.is_enabled())
