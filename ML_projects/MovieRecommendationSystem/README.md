# MovieRecommendationSystem

# Types of Recommendation Systems

**Demographic Filtering**: Recommendations are the same for all users based on broad demographics, e.g., top trending movies.

**Content-Based Filtering**: Recommendations based on movie metadata like genre, cast, or director.

**Collaboration-Based Filtering**: Recommendations based on similar user preferences and behaviors.

## Overview
In today’s world, platforms like YouTube, Netflix, Amazon Prime Video, Hotstar, and many others use recommendation systems to suggest movies or videos based on user preferences. These systems analyze the content watched by the user and recommend movies or shows based on factors like genre, cast, director, and more. This project aims to build a recommendation system that suggests movies based on the user’s watch history.

## Dataset
You can download the movies dataset from [here](Dataset). The dataset contains two CSV files:
1. **Movies Dataset**: Contains details such as movie ID, title, budget, genres, release date, revenue, runtime, and more.
2. **Credits Dataset**: Contains movie-related information like title, cast, and crew.

The dataset columns include:
- **Movies Dataset**: `budget`, `genres`, `id`, `original_language`, `original_title`, `overview`, `popularity`, `production_companies`, `release_date`, `revenue`, `runtime`, `status`, `title`, `vote_average`, `vote_count`.
- **Credits Dataset**: `id`, `title`, `cast`, `crew`.

## Tools and Libraries Used
- **Python**: 3.x
- **Pandas**: 1.2.4
- **Scikit-learn**: 0.24.1

### Install the Libraries
```bash
pip install pandas scikit-learn
