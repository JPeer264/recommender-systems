# Task 1

## Splitted task 1 (Sentence 1 - 3)

crawler - get recent tracks

Fetch API with 500 users and 5000 unique artists

### How?

We loop over the given users and send an API request each. When we receive the data we save the artists in a global array. An `if` statement will check if the minimum amount of artists is reached.

## Splitted task 2 (Sentence 4)

2 files:
- user base (usernames) >> .csv
- histories >> .json
    - suggestion: ein File

Store histories:

```js
history = [
    {
        "user": {
            "id": Int,
            "name": String
        },
        "artist": {
            "id": Int,
            "name": String
        },
        "track": {
            "id": Int,
            "name": String
        },
        "timestamp": Int
    }
];
```

## Splitted task 3 (Sentence 5)

> Data preparation

For example:
- Matching with metadata in a music database
- removing artists and users with less info
- any more?

## Splitted task 4 (Sentence 6)

Save user characteristics of each owner of the fetched histories.

# REPORT

## Splitted task 5 (Sentence 1)

Write a report.

## Splitted task 6 (Sentence 2)

Compare our data with [this](http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_icmr_2016.pdf) data.

- Respect to demographics
    - Country
    - Age
    - Gender

Questions
- Can you make any interesting observations? E.g., is the country, age, or gender distribution different?
- If so, can this be explained by the method you designed for selecting users?
