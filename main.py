from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from logging import info, error, warning, basicConfig, INFO
from multiprocessing import cpu_count, Manager
from functools import lru_cache
from numba import njit
from gmpy2 import mpz
from tqdm import tqdm
import numpy as np
import gmpy2
import math
import time
import os

# Configure logging
basicConfig(filename = "mersenne.log",
            level = INFO,
            format = '%(asctime)s - %(levelname)s - %(message)s')

@lru_cache(maxsize = None)
@njit
def sieve_of_eratosthenes(limit):

    sieve = np.ones(limit // 2, dtype = np.bool8)  # Use 1 bit per entry for memory efficiency
    sieve[0] = False  # 1 is not a prime
    limit_sqrt = int(math.sqrt(limit)) + 1

    for start in range(3, limit_sqrt, 2):

        if sieve[start // 2]:

            for i in range(start * start, limit + 1, start * 2):

                sieve[i // 2] = False

    primes = [2] + [2 * i + 1 for i in range(1, limit // 2) if sieve[i]]
    return primes

def lucas_lehmer_test(p):

    if p == 2: return True  # The smallest Mersenne prime is 2^2 - 1 = 3

    M_p = mpz((1 << p) - 1)  # Efficient power of 2 calculation with gmpy2
    s = mpz(4)  # Lucas-Lehmer seed value

    # Lucas-Lehmer iterations
    for _ in range(p - 2):

        s = (s * s - 2) % M_p

    return s == 0

def process_prime_candidate(p):

    try:

        if lucas_lehmer_test(p):

            M_p = mpz((1 << p) - 1)
            info(f"Mersenne prime found: 2^{p} - 1 = {M_p}")
            return (p, M_p)

    except Exception as e:

        error(f"Error processing p = {p}: {e}")

    return None

def process_batch(prime_batch, shared_mersenne_primes):

    for p in prime_batch:

        if (result := process_prime_candidate(p)):

            shared_mersenne_primes.append(result)

def find_mersenne_primes_parallel(limit, max_workers = None, batch_size = 50):

    if max_workers is None: max_workers = os.cpu_count() or 1  # Use all available CPUs if no specific number is provided.

    prime_candidates = sieve_of_eratosthenes(limit)
    total_batches = (len(prime_candidates) + batch_size - 1) // batch_size

    # Using a Manager to handle shared memory for the Mersenne primes
    with Manager() as manager:

        shared_mersenne_primes = manager.list()  # Shared memory list for primes

        with ProcessPoolExecutor(max_workers = max_workers) as executor:

            futures = {}

            try:

                batches = (prime_candidates[i : i + batch_size] for i in range(0, len(prime_candidates), batch_size))
                futures = {executor.submit(process_batch, batch, shared_mersenne_primes): batch for batch in batches}

                for future in tqdm(as_completed(futures), total = total_batches, desc = "Processing batches"):

                    try:

                        future.result(timeout = 600)  # Increase timeout for larger batches

                    except TimeoutError:

                        warning(f"A batch took too long to complete, skipping batch: {futures[future]}")

                    except Exception as e:

                        error(f"Error during batch processing: {e}")

            except Exception as e:

                error(f"Error in parallel processing: {e}")

            finally:

                executor.shutdown(wait = True)

        # Retrieve the shared primes list
        return list(shared_mersenne_primes)

def validate_positive_int(input_value, default = None):

    try:

        if input_value.strip() == '':  # Default value

            return default

        if (value := int(input_value)) > 0:

            return value

        else:

            error("Input must be a positive integer.")
            return default

    except ValueError:

        error("Invalid input for integer value.")
        return default

if __name__ == "__main__":

    try:

        limit = validate_positive_int(input("Enter the upper p limit when searching for Mersenne primes: "), default = 1000)
        max_workers_input = input(f"Enter the number of parallel workers (cores), or leave empty for default ({cpu_count()} cores): ")
        max_workers = validate_positive_int(max_workers_input, default=cpu_count())

        start_time = time.process_time()
        primes = find_mersenne_primes_parallel(limit, max_workers)
        end_time = time.process_time()

        print("\nMersenne primes found:")
        for p, M_p in primes:

            print(f"2^{p} - 1 = {M_p}")

        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        info(f"Execution completed in {end_time - start_time:.2f} seconds")
        info(f"Mersenne primes found: {primes}")

    except Exception as e:

        error(f"An error occurred: {e}")

# Anieesh Saravanan