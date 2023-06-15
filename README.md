# Nearly-Optimal Hierarchical Clustering for Well-Clustered Graphs
This is the official implementation of the main Algorithm SpecWRSC of the paper "Nearly-Optimal Hierarchical Clustering for Well-Clustered Graphs"

## Requirements
To install the requirements, run the command:
```setup
pip install -r requirements.txt
```

## Evaluation
To evaluate the performance of the algorithm, run the command:
```eval
python eval.py
```

## Results
The results will be displayed in a newly created output file in the ./results/
folder named  "Results_{datetime}.txt", where {datetime} is the current
date and time. For each test, the exact and approximated Dasgupta's cost is printed for the tree constructions considered.
