import numpy as np
import random
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Display the embedding of a random movie"

    def handle(self, *args, **kwargs):
        movies = list(Movie.objects.all())
        movie = random.choice(movies)
        
        embedding_vector = np.frombuffer(movie.emb, dtype=np.float32)
        
        self.stdout.write(f"\nMovie: {movie.title}")
        self.stdout.write(f"Embedding length: {len(embedding_vector)}")
        self.stdout.write(f"First 10 values: {embedding_vector[:10]}")
        self.stdout.write(f"Last 10 values: {embedding_vector[-10:]}")
        