package fr.dvrc.thardy.scheduler;

import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Tuple;
import java.util.Map;

public class EnergyMonitorBolt extends BaseRichBolt {
    private OutputCollector collector;

    @Override
    public void prepare(Map conf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
    }

    @Override
    public void execute(Tuple tuple) {
        String deviceId = tuple.getStringByField("device-id");
        double energy = tuple.getDoubleByField("energy");

        //
        if (energy > 80.0) {
            System.out.println("[ALERT] High Consumption on " + deviceId + ": " + String.format("%.2f", energy) + "W");
        }

        collector.ack(tuple);
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        // No output for now
    }
}