from concurrent.futures import ThreadPoolExecutor, as_completed


def run_parallel(tasks, max_workers=4):
    """
    Run independent callables in parallel.

    tasks = {
        "task_name": callable,
        ...
    }
    """

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fn): name
            for name, fn in tasks.items()
        }

        for future in as_completed(future_map):
            name = future_map[future]
            results[name] = future.result()

    return results
