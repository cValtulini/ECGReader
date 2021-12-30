print("---------------")
print("Testing Library")

import SPxml

file_path = "/content/spxml/xml/example0.xml"
ecg = SPxml.getLeads(file_path)

print("List of leads in ECG:")
for lead in ecg:
    # lead['name']  -  name of ECG lead
    # lead['nsamples'] - number of samples
    # lead['duration'] - total msec of the recording
    # lead['data'] - lead sample data in mv
    print(lead['name'] + " - " + str(lead['nsamples']))
print("---------------")

print(f"type(ecg): {type(ecg)}")
print(f"type(lead): {type(ecg[0])}")
print("---------------")

print("Items in a Lead: ")
for item in ecg[0].items():
  print(item)
print("---------------")
