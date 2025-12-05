import subprocess
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def dump_database(db_config, backup_file_path):
    logging.info(f"Starting database dump for '{db_config['dbname']}' to '{backup_file_path}'...")
    env = os.environ.copy()
    if 'password' in db_config and db_config['password']:
        env['PGPASSWORD'] = db_config['password']

    command = [
        'pg_dump',
        '-h', db_config['host'],
        '-p', str(db_config['port']),
        '-U', db_config['user'],
        '-d', db_config['dbname'],
        '-Fp',
        '-f', backup_file_path
    ]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
        logging.info(f"Database dump completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during database dump. Return code: {e.returncode}")
        logging.error(f"Command executed: {' '.join(command)}")
        logging.error(f"Error output:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logging.error("Error: 'pg_dump' command not found. Make sure PostgreSQL client tools are installed and in your PATH.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during dump: {e}")
        return False
    finally:
        if 'PGPASSWORD' in env:
            del env['PGPASSWORD']


def restore_database(db_config, backup_file_path):
    if not os.path.exists(backup_file_path):
        logging.error(f"Error: Backup file not found at '{backup_file_path}'")
        return False

    logging.info(f"Starting database restore for '{db_config['dbname']}' from '{backup_file_path}'...")
    logging.warning("This function will DROP existing tables before restoring.")

    env = os.environ.copy()
    if 'password' in db_config and db_config['password']:
        env['PGPASSWORD'] = db_config['password']

    tables_to_drop = [
        'relations',
        'properties',
        'urls',
        'cluster_relations',
        'clusters',
        'concepts'
    ]
    drop_commands = " ".join([f"DROP TABLE IF EXISTS {table} CASCADE;" for table in tables_to_drop])
    command_clean = [
        'psql',
        '-h', db_config['host'],
        '-p', str(db_config['port']),
        '-U', db_config['user'],
        '-d', db_config['dbname'],
        '--quiet',
        '-c', drop_commands
    ]

    command_restore = [
        'psql',
        '-h', db_config['host'],
        '-p', str(db_config['port']),
        '-U', db_config['user'],
        '-d', db_config['dbname'],
        '-f', backup_file_path,
        '--quiet',
        '--single-transaction'
    ]

    try:
        logging.info("Step 1/2: Dropping existing tables...")
        clean_process = subprocess.run(command_clean, capture_output=True, text=True, check=True, env=env)
        logging.info("Existing tables dropped (if they existed).")

        logging.info("Step 2/2: Restoring data from backup file...")
        restore_process = subprocess.run(command_restore, capture_output=True, text=True, check=True, env=env)
        logging.info(f"Database restore completed successfully.")

        return True

    except subprocess.CalledProcessError as e:
        if e.cmd[0] == 'psql' and '-c' in e.cmd:
            logging.error("Error during pre-restore cleaning step.")
        else:
            logging.error("Error during database restore step.")

        logging.error(f"Error during database operation. Return code: {e.returncode}")
        logging.error(f"Command executed: {' '.join(e.cmd)}")
        logging.error(f"Error output:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logging.error(
            "Error: 'psql' command not found. Make sure PostgreSQL client tools are installed and in your PATH.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during restore: {e}")
        return False
    finally:
        if 'PGPASSWORD' in env:
            env.pop('PGPASSWORD', None)


if __name__ == "__main__":
    try:
        with open('../config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        db_connection_info = config['database']
    except FileNotFoundError:
        logging.error("config.yaml not found. Cannot run example usage.")
        exit(1)
    except KeyError:
         logging.error("Could not find 'database' section in config.yaml.")
         exit(1)
    except Exception as e:
        logging.error(f"Error loading config.yaml: {e}")
        exit(1)


    backup_file = '../.cache/ontology_dbpedia_backup.sql'

    logging.info("Database Dump")
    if dump_database(db_connection_info, backup_file):
        logging.info(f"Dump successful. Backup saved to {backup_file}")

        confirm = input(f"\nWARNING\nThis will attempt to restore '{backup_file}' into the database '{db_connection_info['dbname']}' on host '{db_connection_info['host']}'.\n"
                        "This may OVERWRITE existing data.\n"
                        "Are you sure you want to proceed? (y/n): ").lower()

        if confirm == 'y':
            logging.info("\nDatabase Restore")
            if restore_database(db_connection_info, backup_file):
                logging.info("Restore successful.")
            else:
                logging.error("Restore failed.")
        else:
            logging.info("Restore operation cancelled by user.")
    else:
        logging.error("Dump failed. Restore will not be attempted.")
