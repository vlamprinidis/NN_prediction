# evt.key,  # Name
# # Self CPU total %
# format_time_share(evt.self_cpu_time_total,
#                   self_cpu_time_total),
# evt.self_cpu_time_total_str,  # Self CPU total
# # CPU total %
# format_time_share(evt.cpu_time_total, self_cpu_time_total),
# evt.cpu_time_total_str,  # CPU total
# evt.cpu_time_str,  # CPU time avg

# evt.self_cpu_time_total,  # Self CPU total
# evt.count,  # Number of calls
# evt.cpu_time,  # CPU time avg
# evt.cpu_time_total

def give_evt_lst(evt):
    return [evt.key, str(evt.cpu_time), 
            str(evt.cpu_time_total), str(evt.count)]

# give_evt_lst(prof.function_events[10])

import csv
from statistics import mean

def save_to_csv(function_events, name):
    headers = [
        'Name',
        'CPU time avg (us)',
        'CPU total (us)',
        'Number of Calls'
    ]
    with open(name, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        #way to write to csv file
        writer.writerow(headers)
        for evt in function_events:
            writer.writerow( give_evt_lst(evt) )
            
def avg_2nodes(file1,file2,result_name):
    from csvsort import csvsort
    csvsort(file1, [0])
    csvsort(file2, [0])
    with open(file1) as csv1, open(file2) as csv2, open(result_name, mode='w') as csv_res:
        read1 = csv.reader(csv1)
        read2 = csv.reader(csv2)
        writer = csv.writer(csv_res)
        
        headers = read1[0]
        writer.writerow(headers)
        for row1,row2 in zip(read1[1:],read2[1:]):
            op = row1[0]
            new_row = [ op ] + [ mean([x,y]) for x,y in zip(row1[1:],row2[1:]) ]
            writer.writerow(new_row)
            