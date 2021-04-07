import datetime


def datetime2str(datetime_obj):
    """ 
        Receives a datetiem object and returns a string
        in format 'day/month/year hr:mins:secs' 
    """
    
    return datetime_obj.strftime('%d/%m/%Y %H:%M:%S')

def str2datetime(date_str):
    """
        Receives a string with format 'day/month/year hr:mins:secs'
        and returns a datetime object
    """
    date, time = date_str.split()
    day, month, year = date.split('/')
    hr, mins, secs = time.split(':')
    return datetime.datetime(int(year), int(month), int(day), 
                             int(hr), int(mins), int(secs)
                            )
    
