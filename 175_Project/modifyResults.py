import matplotlib.pyplot as plt
file1 = open("returnsRainbowOriginal.txt","r") 
count = 0

l = []
while True: 
    count += 1
    
    # Get next line from file 
    line = file1.readline() 
    # if line is empty 
    # end of file is reached 
    if not line: 
        break    
    words = line.split()
    
    newWords = [float(i) for i in words]
    
    if (newWords[1] > 50.0):
        newWords[1] += 100
    
    l.append(newWords)
    
    # Comment the above segment out and uncomment the below if you want to run the below snippet
    # Rename the file to write to something like returnsRainbowModified-EpisodeNumberVsReturn
    '''
    if (newWords[1] > 50.0):
        newWords[1] += 100
    else:
        newWords[1]  = 0
    l.append(newWords)   
    '''
    
    
    
x = []
y = []

for elem in l:
    x.append(elem[0])
    y.append(elem[1])
plt.plot(x, y)
plt.show()
plt.title('Episode vs Return')
plt.ylabel('Return')
plt.xlabel('Episode')
plt.savefig('returnsMODIFIEDEPISODE.png')        # broken, ignore this






print(l)
with open('returnsRainbowModified-EpisodeNumberVsReturn.txt', 'w') as f:
    for item in l:
        f.write("{}\t{}\n".format(item[0], item[1]))

file1.close() 