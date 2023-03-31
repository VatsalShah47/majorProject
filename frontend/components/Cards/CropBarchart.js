import React from "react";
import Chart from "chart.js";
import { useState,useEffect } from "react";

export default function CardBarChart() {
  const [chartData, setChartData] = useState({
    "labels":[],
    "datasets":[],
    "xlabel":"xlabel",
    "ylabel":"ylabel"
  });
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/crop-predict");
        const data = await response.json();
        setChartData(data);
        console.log(data)
      } catch (error) {
        console.log(error);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    // Get the canvas element
    const ctx = document.getElementById("myChart3").getContext("2d");

    // Create the chart
    const myChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: chartData["labels"],
        datasets: [
          {
            label: "Frequency",
            data: chartData["datasets"],
            backgroundColor: "rgba(54, 162, 235, 0.2)",
            borderColor: "rgba(54, 162, 235, 1)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        scales: {
          yAxes: [
            {
              ticks: {
                beginAtZero: true,
              },
            },
          ],
        },
      },
    });
  }, [chartData]);

  return (
    <canvas id="myChart3"></canvas>
  );
}

