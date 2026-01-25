package fr.dvrc.thardy.scheduler;

import org.apache.storm.spout.SpoutOutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichSpout;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;
import java.util.Map;
import java.util.Random;


public class SensorSpout extends BaseRichSpout {

    private SpoutOutputCollector collector;
    private Random rand;

    @Override
    public void open(Map conf, TopologyContext context, SpoutOutputCollector collector) {
        this.collector = collector;
        this.rand = new Random();
    }

    @Override
    public void nextTuple() {
        // Simulate energy usage data from an edge device
        double energyUsage = rand.nextDouble() * 100; // 0 to 100 watts
        String deviceId = "edge-node-" + rand.nextInt(10);

        // Emit: [DeviceID, EnergyValue]
        collector.emit(new Values(deviceId, energyUsage));

        // Slow down simulation
        try { Thread.sleep(100); } catch (InterruptedException e) {}
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("device-id", "energy"));
    }
}