import gzip

def save_first_n_lines_cif(input_gz, output_txt, n):
    """
    Reads the first n lines from a .cif.gz file and saves them to a .txt file.
    """
    try:
        # Open the .gz file in read-text mode ('rt')
        with gzip.open(input_gz, 'rt') as f_in:
            # Open the output file in write mode ('w')
            with open(output_txt, 'w') as f_out:
                for i, line in enumerate(f_in):
                    if i >= n:
                        break
                    f_out.write(line)
        print(f"Successfully saved first {n} lines to {output_txt}")
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    n = 500_000
    save_first_n_lines_cif('data/pdb/raw/ccd/components.cif.gz', f'data/pdb/raw/ccd/components_{n}_lines.txt', n)
