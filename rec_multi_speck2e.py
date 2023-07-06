import samna, samnagui
from multiprocessing import Process
import time, socket

def get_free_tcp_port():
    """Returns a free tcp port.
    Returns:
        str: A port which is free in the system
    """
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(("0.0.0.0", 0))
    free_socket.listen(5)
    port = free_socket.getsockname()[1]  # get port
    free_socket.close()
    return port


def record(file_name: str, t: int = 5):
    # set up graphs
    save_graphs = []
    buffers = []
    for sensor_idx, handle in enumerate(handles): # global handles
        buf = samna.BasicSinkNode_speck2e_event_output_event() 
        graph = samna.graph.EventFilterGraph()
        graph.sequential([
            handle.get_model_source_node(),
            buf
        ])
        buffers.append(buf)
        save_graphs.append(graph)
        stop_watch = handle.get_stop_watch()
        stop_watch.set_enable_value(False)
        stop_watch.set_enable_value(True)
        graph.start()
    print("Started recording!")
    time.sleep(t)
    print("Stopped recording!")
    for sensor_idx, buf in enumerate(buffers):
        write_file_name = f"{file_name}_sensor_{sensor_idx}.bin"
        print(f"Writing to file: {write_file_name}")
        # create file
        binary_file = open(write_file_name, "wb")
        # write events
        for e in buf.get_events():
            if isinstance(e, samna.speck2e.event.Spike):
                # print(e.x, e.y, e.timestamp, e.feature)
                binary_file.write((e.x).to_bytes(4, byteorder = 'little'))
                binary_file.write((e.y).to_bytes(4, byteorder = 'little'))
                binary_file.write((e.feature).to_bytes(4, byteorder = 'little'))
                binary_file.write((e.timestamp).to_bytes(4, byteorder = 'little'))
        # close file
        binary_file.close()
    # close graph without leaks
    for graph in save_graphs:
        graph.stop()
    for buf in buffers:
        buf.get_events()


# constants
endpoint_prefix = "tcp://0.0.0.0:"

# get devices
devs = samna.device.get_unopened_devices()
handles = [samna.device.open_device(dev) for dev in devs]

# graphs
graphs = []
graph_members = []
endpoint_ports = []
process_handles = []

for handle in handles:
    default_config = samna.speck2e.configuration.SpeckConfiguration()
    default_config.dvs_layer.monitor_enable = True
    # default_config.dvs_layer.raw_monitor_enable = True
    # default_config.dvs_layer.pass_sensor_events = False
    handle.get_model().apply_configuration(default_config)

for sensor_idx, handle in enumerate(handles):
    port = get_free_tcp_port()
    visualizer_process = Process(
        target=samnagui.run_visualizer,
        args=(endpoint_prefix + str(port), 0.75, 0.75)
    )
    process_handles.append(visualizer_process)
    visualizer_process.start()
    graph = samna.graph.EventFilterGraph() 
    members = graph.sequential(
        [
            handle.get_model_source_node(),
            "Speck2eDvsToVizConverter",
            "VizEventStreamer"
        ]
    )
    members[2].set_streamer_destination(endpoint_prefix + str(port))
    if members[2].wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {endpoint_prefix + str(port)} fails')
    graphs.append(graph)
    graph_members.append(members)
    endpoint_ports.append(port)

    visualizer_config = samna.ui.VisualizerConfiguration(
        plots=[
            samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer", [0, 0, 1, 1])
        ]
    )
    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), members[2]])
    config_source.write([visualizer_config])
    graph.start()
    print(f"Sensor {sensor_idx} started!")
