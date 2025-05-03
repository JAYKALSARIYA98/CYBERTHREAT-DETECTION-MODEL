import pandas as pd
import numpy as np
import joblib
import pyshark
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scapy.all import rdpcap, IP
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def extract_features_from_pcap(pcap_file):
    """Extract features from pcap file to match the UNSW-NB15 dataset structure"""
    print(f"Analyzing file: {pcap_file}")
    
    # Use pyshark to read the pcapng/pcap file
    try:
        capture = pyshark.FileCapture(pcap_file)
    except Exception as e:
        print(f"Error opening capture file with pyshark: {e}")
        print("Trying with scapy...")
        try:
            packets = rdpcap(pcap_file)
        except Exception as e:
            print(f"Error opening capture file with scapy: {e}")
            return None
    
    # Initialize a list to store flow data
    flows = defaultdict(lambda: {
        'spkts': 0, 'dpkts': 0,
        'sbytes': 0, 'dbytes': 0,
        'sttl': 0, 'dttl': 0,
        'proto': '',
        'service': '-',
        'state': '-',
        'dur': 0,
        'rate': 0,
        'sload': 0, 'dload': 0,
        'sloss': 0, 'dloss': 0,
        'sinpkt': 0, 'dinpkt': 0,
        'sjit': 0, 'djit': 0,
        'swin': 0, 'dwin': 0,
        'stcpb': 0, 'dtcpb': 0,
        'tcprtt': 0, 'synack': 0, 'ackdat': 0,
        'smean': 0, 'dmean': 0,
        'trans_depth': 0, 'response_body_len': 0,
        'ct_srv_src': 0, 'ct_state_ttl': 0,
        'ct_dst_ltm': 0, 'ct_src_dport_ltm': 0,
        'ct_dst_sport_ltm': 0, 'ct_dst_src_ltm': 0,
        'is_ftp_login': 0, 'ct_ftp_cmd': 0,
        'ct_flw_http_mthd': 0, 'ct_src_ltm': 0,
        'ct_srv_dst': 0, 'is_sm_ips_ports': 0,
        'start_time': None, 'end_time': None,
        'src_ip': '', 'dst_ip': '', 'src_port': '', 'dst_port': ''  # Store IPs and ports for visualization
    })
    
    # Track connections for counting purposes
    connections = defaultdict(int)
    
    try:
        # First pass to collect basic information
        print("Collecting basic flow information...")
        for i, packet in enumerate(capture if 'capture' in locals() else packets):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i} packets...")
            
            try:
                if 'capture' in locals():  # pyshark
                    if 'ip' not in packet:
                        continue
                    
                    src_ip = packet.ip.src
                    dst_ip = packet.ip.dst
                    
                    try:
                        proto = packet.transport_layer.lower() if hasattr(packet, 'transport_layer') else 'other'
                    except:
                        proto = 'other'
                    
                    try:
                        src_port = packet[proto].srcport if hasattr(packet, proto) and hasattr(packet[proto], 'srcport') else '0'
                        dst_port = packet[proto].dstport if hasattr(packet, proto) and hasattr(packet[proto], 'dstport') else '0'
                    except:
                        src_port = '0'
                        dst_port = '0'
                    
                    length = int(packet.length)
                    
                    try:
                        ttl = int(packet.ip.ttl)
                    except:
                        ttl = 0
                    
                else:  # scapy
                    if not packet.haslayer(IP):
                        continue
                    
                    src_ip = packet[IP].src
                    dst_ip = packet[IP].dst
                    proto = packet[IP].proto
                    
                    # Map protocol numbers to names
                    proto_map = {6: 'tcp', 17: 'udp', 1: 'icmp'}
                    proto = proto_map.get(proto, str(proto))
                    
                    try:
                        src_port = str(packet.sport) if hasattr(packet, 'sport') else '0'
                        dst_port = str(packet.dport) if hasattr(packet, 'dport') else '0'
                    except:
                        src_port = '0'
                        dst_port = '0'
                    
                    length = len(packet)
                    ttl = packet[IP].ttl
                
                # Create a flow key
                forward_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
                backward_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{proto}"
                
                # Get current timestamp
                timestamp = time.time()
                
                # Check if it's a new flow or existing flow
                if forward_key in flows:
                    key = forward_key
                    flows[key]['spkts'] += 1
                    flows[key]['sbytes'] += length
                    flows[key]['sttl'] = ttl  # Update TTL
                    flows[key]['end_time'] = timestamp
                elif backward_key in flows:
                    key = backward_key
                    flows[key]['dpkts'] += 1
                    flows[key]['dbytes'] += length
                    flows[key]['dttl'] = ttl  # Update TTL
                    flows[key]['end_time'] = timestamp
                else:
                    key = forward_key
                    flows[key]['proto'] = proto
                    flows[key]['spkts'] = 1
                    flows[key]['sbytes'] = length
                    flows[key]['sttl'] = ttl
                    flows[key]['src_ip'] = src_ip
                    flows[key]['dst_ip'] = dst_ip
                    flows[key]['src_port'] = src_port
                    flows[key]['dst_port'] = dst_port
                    flows[key]['start_time'] = timestamp
                    flows[key]['end_time'] = timestamp
                
                # Update connection tracking
                connections[f"{src_ip}-{dst_ip}"] += 1
                
                # Determine state based on flags for TCP
                if proto == 'tcp':
                    if 'capture' in locals():  # pyshark
                        if hasattr(packet.tcp, 'flags'):
                            flags = packet.tcp.flags
                            if '0x0002' in flags:  # SYN flag
                                flows[key]['state'] = 'SYN'
                            elif '0x0011' in flags:  # FIN and ACK flags
                                flows[key]['state'] = 'FIN'
                            elif '0x0010' in flags:  # ACK flag
                                flows[key]['state'] = 'ACK'
                            elif '0x0004' in flags:  # RST flag
                                flows[key]['state'] = 'RST'
                    else:  # scapy
                        if hasattr(packet, 'flags'):
                            if packet.flags & 0x02:  # SYN
                                flows[key]['state'] = 'SYN'
                            elif packet.flags & 0x01 and packet.flags & 0x10:  # FIN and ACK
                                flows[key]['state'] = 'FIN'
                            elif packet.flags & 0x10:  # ACK
                                flows[key]['state'] = 'ACK'
                            elif packet.flags & 0x04:  # RST
                                flows[key]['state'] = 'RST'
                
            except Exception as e:
                print(f"Error processing packet {i}: {e}")
                continue
                
        print("Calculating derived features...")
        # Calculate derived features
        for key, flow in flows.items():
            if flow['end_time'] and flow['start_time']:
                flow['dur'] = flow['end_time'] - flow['start_time']
                if flow['dur'] > 0:
                    flow['rate'] = (flow['spkts'] + flow['dpkts']) / flow['dur']
                    if flow['spkts'] > 0:
                        flow['sinpkt'] = flow['dur'] / flow['spkts']
                    if flow['dpkts'] > 0:
                        flow['dinpkt'] = flow['dur'] / flow['dpkts']
            
            if flow['dur'] > 0:
                flow['sload'] = flow['sbytes'] * 8 / flow['dur'] if flow['dur'] > 0 else 0
                flow['dload'] = flow['dbytes'] * 8 / flow['dur'] if flow['dur'] > 0 else 0
            
            flow['smean'] = flow['sbytes'] / flow['spkts'] if flow['spkts'] > 0 else 0
            flow['dmean'] = flow['dbytes'] / flow['dpkts'] if flow['dpkts'] > 0 else 0
            
            # Count connection-related features based on our simplified tracking
            src_ip, src_port = key.split('-')[0].split(':')
            dst_ip, dst_port = key.split('-')[1].split(':')
            
            flow['ct_srv_src'] = sum(1 for k in connections.keys() if k.startswith(f"{src_ip}-"))
            flow['ct_dst_ltm'] = sum(1 for k in connections.keys() if k.endswith(f"-{dst_ip}"))
            
        
        # Convert to DataFrame
        df_flows = pd.DataFrame.from_dict(flows, orient='index').reset_index(drop=True)
        
        # Add id column
        df_flows.insert(0, 'id', range(1, len(df_flows) + 1))
        
        # Make sure all required features are present
        required_features = [
            'id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 
            'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
            'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 
            'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 
            'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 
            'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
            'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 
            'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
        ]
        
        for feature in required_features:
            if feature not in df_flows.columns:
                df_flows[feature] = 0
        
        return df_flows
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
    finally:
        if 'capture' in locals():
            capture.close()

def predict_attacks(features_df, binary_model, multiclass_model):
    """Predict if packets are attacks and classify them"""
    if features_df is None or len(features_df) == 0:
        print("No valid features extracted for prediction.")
        return None
    
    # Prepare features for prediction
    X = features_df.drop(['id', 'src_ip', 'dst_ip', 'src_port', 'dst_port'], axis=1, errors='ignore')
    
    # Save original features for adding back later
    original_features = features_df[['id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'proto']]
    
    # Binary prediction (attack or normal)
    print("\nPredicting attack presence...")
    binary_predictions = binary_model.predict(X)
    binary_proba = binary_model.predict_proba(X)[:, 1]  # Probability of being an attack
    
    # Add predictions to dataframe
    features_df['is_attack'] = binary_predictions
    features_df['attack_probability'] = binary_proba
    
    # For flows classified as attacks, predict the attack type
    attack_indices = features_df[features_df['is_attack'] == 1].index
    
    if len(attack_indices) > 0:
        print(f"Found {len(attack_indices)} potential attack flows.")
        attack_features = X.iloc[attack_indices]
        attack_types = multiclass_model.predict(attack_features)
        
        # Initialize attack_type column with 'Normal'
        features_df['attack_type'] = 'Normal'
        
        # Update attack types for detected attacks
        features_df.loc[attack_indices, 'attack_type'] = attack_types
    else:
        print("No attacks detected.")
        features_df['attack_type'] = 'Normal'
    
    return features_df

def visualize_results(results_df, output_prefix):
    """Generate visualizations from the analysis results"""
    if results_df is None or len(results_df) == 0:
        print("No data available for visualization.")
        return
    
    # Filter only attack packets
    attack_df = results_df[results_df['is_attack'] == 1].copy()
    
    if len(attack_df) == 0:
        print("No attacks detected for visualization.")
        return
    
    print("\nGenerating visualizations...")
    
    # 1. Attack type distribution
    plt.figure(figsize=(12, 6))
    attack_counts = attack_df['attack_type'].value_counts()
    sns.barplot(x=attack_counts.index, y=attack_counts.values)
    plt.title('Distribution of Detected Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/{output_prefix}_attack_distribution.png')
    plt.close()
    
    # 2. Attack probability distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(attack_df['attack_probability'], bins=20, kde=True)
    plt.title('Distribution of Attack Probability Scores')
    plt.xlabel('Attack Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'results/{output_prefix}_attack_probability.png')
    plt.close()
    
    # 3. Protocol distribution in attacks
    plt.figure(figsize=(10, 6))
    proto_counts = attack_df['proto'].value_counts()
    sns.barplot(x=proto_counts.index, y=proto_counts.values)
    plt.title('Protocols Used in Attacks')
    plt.xlabel('Protocol')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'results/{output_prefix}_attack_protocols.png')
    plt.close()
    
    # 4. Attack type by protocol
    plt.figure(figsize=(14, 8))
    attack_proto = pd.crosstab(attack_df['attack_type'], attack_df['proto'])
    attack_proto.plot(kind='bar', stacked=True)
    plt.title('Attack Types by Protocol')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Protocol')
    plt.tight_layout()
    plt.savefig(f'results/{output_prefix}_attack_by_protocol.png')
    plt.close()
    
    # 5. Most targeted destinations
    plt.figure(figsize=(12, 6))
    dst_counts = attack_df['dst_ip'].value_counts().head(10)
    sns.barplot(x=dst_counts.index, y=dst_counts.values)
    plt.title('Top 10 Targeted Destination IPs')
    plt.xlabel('Destination IP')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/{output_prefix}_targeted_destinations.png')
    plt.close()
    
    # 6. Source of attacks
    plt.figure(figsize=(12, 6))
    src_counts = attack_df['src_ip'].value_counts().head(10)
    sns.barplot(x=src_counts.index, y=src_counts.values)
    plt.title('Top 10 Attack Source IPs')
    plt.xlabel('Source IP')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/{output_prefix}_attack_sources.png')
    plt.close()
    
    print(f"Visualizations saved to results/{output_prefix}_*.png")

def summarize_results(results_df):
    """Summarize the attack detection results"""
    if results_df is None:
        return "No results to summarize."
    
    total_flows = len(results_df)
    attack_flows = len(results_df[results_df['is_attack'] == 1])
    attack_percentage = (attack_flows / total_flows) * 100 if total_flows > 0 else 0
    
    print("\n===== Network Traffic Analysis Results =====")
    print(f"Total flows analyzed: {total_flows}")
    print(f"Flows classified as attacks: {attack_flows} ({attack_percentage:.2f}%)")
    
    if attack_flows > 0:
        attack_types = results_df[results_df['is_attack'] == 1]['attack_type'].value_counts()
        print("\nAttack Categories Detected:")
        for attack_type, count in attack_types.items():
            print(f"  - {attack_type}: {count} flows ({(count/attack_flows)*100:.2f}% of attacks)")
        
        # Get the most suspicious flows
        suspicious_flows = results_df.sort_values('attack_probability', ascending=False).head(5)
        print("\nMost Suspicious Flows:")
        for idx, row in suspicious_flows.iterrows():
            print(f"  - Flow ID: {row['id']}, Protocol: {row['proto']}, Source: {row['src_ip']}:{row['src_port']}, " + 
                  f"Destination: {row['dst_ip']}:{row['dst_port']}, Attack Type: {row['attack_type']}, " + 
                  f"Confidence: {row['attack_probability']:.2f}")
    
    return "Analysis complete. " + ("Attacks detected!" if attack_flows > 0 else "No attacks detected.")

def save_attack_results(results_df, output_file):
    """Save only attack results to a CSV file"""
    if results_df is None or len(results_df) == 0:
        print("No results to save.")
        return
    
    # Filter only attack packets
    attack_df = results_df[results_df['is_attack'] == 1].copy()
    
    if len(attack_df) == 0:
        print("No attacks detected to save.")
        return
    
    # Select relevant columns
    columns_to_save = [
        'id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'proto',
        'state', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes',
        'rate', 'attack_type', 'attack_probability'
    ]
    
    # Save to CSV
    attack_df[columns_to_save].to_csv(output_file, index=False)
    print(f"Attack results saved to {output_file}")
    
    return attack_df

def main():
    parser = argparse.ArgumentParser(description="Analyze network traffic from pcap files for attacks")
    parser.add_argument("pcap_file", help="Path to the pcap/pcapng file to analyze")
    parser.add_argument("--models_dir", default="models", help="Directory containing trained models")
    parser.add_argument("--output", help="Custom output prefix for result files")
    args = parser.parse_args()
    
    # Define output prefix based on pcap filename or custom output
    if args.output:
        output_prefix = args.output
    else:
        # Extract filename without extension
        output_prefix = os.path.splitext(os.path.basename(args.pcap_file))[0]
    
    # Check if pcap file exists
    if not os.path.exists(args.pcap_file):
        print(f"Error: File {args.pcap_file} does not exist")
        return
    
    # Check if models exist
    binary_model_path = os.path.join(args.models_dir, "binary_attack_model.pkl")
    multiclass_model_path = os.path.join(args.models_dir, "multiclass_attack_model.pkl")
    
    if not os.path.exists(binary_model_path) or not os.path.exists(multiclass_model_path):
        print(f"Error: Models not found in {args.models_dir}. Please train the models first.")
        return
    
    # Load models
    print("Loading models...")
    binary_model = joblib.load(binary_model_path)
    multiclass_model = joblib.load(multiclass_model_path)
    
    # Extract features from pcap
    features_df = extract_features_from_pcap(args.pcap_file)
    
    if features_df is not None and not features_df.empty:
        # Predict attacks
        results = predict_attacks(features_df, binary_model, multiclass_model)
        
        # Summarize results
        conclusion = summarize_results(results)
        print(f"\nConclusion: {conclusion}")
        
        # Save attack results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"results/{output_prefix}_attacks_{timestamp}.csv"
        save_attack_results(results, output_csv)
        
        # Generate visualizations
        visualize_results(results, output_prefix)
    else:
        print("Failed to extract features from the pcap file.")

if __name__ == "__main__":
    main() 