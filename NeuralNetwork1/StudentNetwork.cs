using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private readonly double[][,] Weights;
        private readonly double[][] Charges;
        private readonly double[][] Errors; 
        private readonly Stopwatch Stop_watch = new Stopwatch();

        public StudentNetwork(int[] structure, double lowerBound = -1, double upperBound = 1)
        {
            Charges = new double[structure.Length][];
            Errors = new double[structure.Length][];
            
            for (int i = 0; i < structure.Length; i++)
            {
                Errors[i] = new double[structure[i]];
                Charges[i] = new double[structure[i] + 1];
                Charges[i][structure[i]] = 1;
            }
            
            Weights = new double[structure.Length - 1][,];
            
            for(int n = 0; n < structure.Length - 1; n++)
            {
                var r = new Random();
                var rowsCount = structure[n] + 1;
                var columnsCount = structure[n + 1];
                
                Weights[n] = new double[rowsCount, columnsCount];

                for (int i = 0; i < rowsCount; i++)
                {
                    for (int j = 0; j < columnsCount; j++)
                    {
                        Weights[n][i, j] = lowerBound + r.NextDouble() * (upperBound - lowerBound);
                    }
                }
            }
        }

        
        private double Quadratic_error(double[] output)
        {
            double res = 0;

            for (int i = 0; i < output.Length; i++)
            {
                res += Math.Pow(output[i] - Charges[Charges.Length - 1][i], 2);
            }    
            res /= output.Length;
            return res;
        }

        
        private void Run(double[] input)
        {
            for (int j = 0; j < input.Length; j++) 
                Charges[0][j] = input[j];
            
            for (int i = 1; i < Charges.GetLength(0); i++)
                Coming_Out_to_next_layer(Charges[i - 1], Weights[i - 1], Charges[i]);
        }

       
        private static void Coming_Out_to_next_layer(double[] vector, double [,] matrix, double[] result)
        {
            var rowsCount = matrix.GetLength(0);
            var colCount = matrix.GetLength(1);
            
            for (int i = 0; i < colCount ; i++)
            {
                double sum = 0;
                
                for(int j = 0; j < rowsCount; j++) 
                    sum += vector[j] * matrix[j, i];
                
                result[i] = Activation_function_Sigmoid(sum);
            }  
        }

        
        private static double Activation_function_Sigmoid(double value) => 1.0 / (Math.Exp(-value) + 1);

        
        private void Back_Propagation(double[] output)
        {
            
            for (var j = 0; j < output.Length; j++)
            {
                var currentCharge = Charges[Errors.Length - 1][j];
                var expectedOutput = output[j];
                Errors[Errors.Length - 1][j] = currentCharge * (1 - currentCharge) * (expectedOutput - currentCharge);
            }

            for (int i = Errors.Length - 2; i >= 1; i--)
            {
                for (int j = 0; j < Errors[i].Length; j++)
                {
                    var charge = Charges[i][j] * (1 - Charges[i][j]);
                    var scul = 0.0;
                    for (int k = 0; k < Errors[i + 1].Length; k++)
                        scul += Errors[i + 1][k] * Weights[i][j, k];
                    Errors[i][j] = charge * scul;
                }
            }

            for (int n = 0; n < Weights.Length; n++)
            {
                for (int i = 0; i < Weights[n].GetLength(0); i++)
                {
                    for (int j = 0; j < Weights[n].GetLength(1); j++)
                    {
                        var DW = 0.25 * Errors[n + 1][j] * Charges[n][i];
                        Weights[n][i, j] += DW;
                    }
                }
            }

        }
       
       

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int i = 1;
            while (sample.EstimatedError() > acceptableError)
            {
                Run(sample.input);
                Back_Propagation(sample.Output);
                ++i;
            }

            return i;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            // Конструируем массивы входов и выходов
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            // Группируем массивы из samplesSet в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            int epoch_to_run = 0;
            double error = double.PositiveInfinity;

            Stop_watch.Restart();

            while (epoch_to_run < epochsCount && error > acceptableError)
            {
                epoch_to_run++;
                error = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    Run(inputs[i]);
                    Back_Propagation(outputs[i]);
                    error += Quadratic_error(outputs[i]);
                }
                error /= inputs.Length;

                OnTrainProgress((epoch_to_run * 1.0) / epochsCount, error, Stop_watch.Elapsed);
            }
            OnTrainProgress(1, error, Stop_watch.Elapsed);
            Stop_watch.Stop();
            return error;
        }

        protected override double[] Compute(double[] input)
        {
            Run(input);
            return Charges.Last().Take(Charges.Last().Length - 1).ToArray();
        }
    }
}