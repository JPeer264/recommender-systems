# HOW TO RUN A SUCCESSFUL TEST

## First

Change Method Name before every single run

Name equals "Name of recommender"+"Name of AAM"<br>
e.g.
```python
METHOD = "HR_RB_WIKI"
```

## Second

Change number of Artists before every change between WIKI or LYRICS<br>
WIKI = 10119<br>
LYRICS = 3000<br>
e.g.
```python
MAX_ARTIST = 3000
```

## Third

Check if you cut the UAM and the u_adix!<br>
inside main method
```python
UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)[:, :MAX_ARTISTS]
```
inside run method
```python
u_aidx = np.nonzero(UAM[u, :MAX_ARTISTS])[0]
```

## Fourth (just with AAM)

The line where you read the AAM file check if it looks like this (look at the ending):

```python
AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)[:MAX_ARTISTS, :MAX_ARTISTS]
```
